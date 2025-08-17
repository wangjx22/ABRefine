"""Code."""
import numpy as np
import scipy
from scipy.spatial import distance_matrix

import np.residue_constants as rc


def extract_tip_coordinates(po):
    """Run extract_tip_coordinates method."""
    # code.
    coordinates = []
    for i, aa in enumerate(po.aatype):
        resname = rc.restype_1to3[rc.restype_order_with_x_inv[aa]]
        atom_name = rc.AA_to_tip[resname]
        atom_ix = rc.atom_order[atom_name]
        if po.atom_mask[i, atom_ix]:
            atom_coord = po.atom_positions[i, atom_ix]
            coordinates.append(atom_coord)
        else:
            coordinates.append(po.atom_positions[i, rc.atom_order['CA']])
    return np.array(coordinates)


def get_coords(po):
    """Run get_coords method."""
    # code.
    N = po.atom_positions[:, 0]
    Ca = po.atom_positions[:, 1]
    C = po.atom_positions[:, 2]
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = ca * a + cb * b + cc * c
    return N, Ca, C, Ca + Cb


def set_lframe(pdict):
    """Run set_lframe method."""
    # code.
    """
    set_lframe:
    Args:
        pdict : pdict
    Returns:
    """
    z = pdict['Cb'] - pdict['Ca']
    z /= np.linalg.norm(z, axis=-1)[:, (None)]
    x = np.cross(pdict['Ca'] - pdict['N'], z)
    x /= np.linalg.norm(x, axis=-1)[:, (None)]    # TODO: maybe divide by 0
    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1)[:, (None)]
    xyz = np.stack([x, y, z])
    pdict['lfr'] = np.transpose(xyz, [1, 0, 2])


def get_dihedrals(a, b, c, d):
    """Run get_dihedrals method."""
    # code.
    """
    get_dihedrals:
    Args:
        a : a
        b : b
        c : c
        d : d
    Returns:
    """
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1, axis=-1)[:, (None)]
    v = b0 - np.sum(b0 * b1, axis=-1)[:, (None)] * b1
    w = b2 - np.sum(b2 * b1, axis=-1)[:, (None)] * b1
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def get_angles(a, b, c):
    """Run get_angles method."""
    # code.
    """
    get_angles:
    Args:
        a : a
        b : b
        c : c
    Returns:
    """
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, (None)]
    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, (None)]
    x = np.sum(v * w, axis=1)
    return np.arccos(x)


def set_neighbors6D(pdict, aa_features):
    """Run set_neighbors6D method."""
    # code.
    """
    set_neighbors6D:
    Args:
        pdict : pdict
    Returns:
    """
    N = pdict['N']
    Ca = pdict['Ca']
    Cb = pdict['Cb']
    nres = pdict['Ca'].shape[0]
    dmax = 20.0
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)
    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]
    # dist6d = np.zeros((nres, nres))
    # dist6d[idx0, idx1] = np.linalg.norm(Cb[idx1] - Cb[idx0], axis=-1)
    # pdict['dist6d'] = dist6d
    if 'omega6d' in aa_features:
        omega6d = np.zeros((nres, nres))
        omega6d[idx0, idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])
        pdict['omega6d'] = omega6d.astype(np.float32)
    if 'theta6d' in aa_features:
        theta6d = np.zeros((nres, nres))
        theta6d[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
        pdict['theta6d'] = theta6d.astype(np.float32)
    if 'phi6d' in aa_features:
        phi6d = np.zeros((nres, nres))
        phi6d[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
        pdict['phi6d'] = phi6d.astype(np.float32)


def dihedral_angle(a, b, c, d):
    """Run dihedral_angle method."""
    # code.
    """
    dihedral_angle:
    Args:
        a : a
        b : b
        c : c
        d : d
    Returns:
    """
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c
    norm = np.linalg.norm(b1)
    if norm == 0:
        return -100
    b1 /= norm    # TODO: maybe divide by 0
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.arctan2(y, x)


def compute_psi_phi(coordinates):
    """Run compute_psi_phi method."""
    # code.
    """
    compute_psi_phi:
    Args:
        coordinates : coordinates
    Returns:
    """
    N = coordinates['N']
    CA = coordinates['Ca']
    C = coordinates['C']
    phi = [0]
    psi = []
    for i in range(0, len(N) - 1):
        psi_angle = dihedral_angle(N[i], CA[i], C[i], N[i + 1])
        psi.append(psi_angle)
    for i in range(1, len(N)):
        phi_angle = dihedral_angle(C[i - 1], N[i], CA[i], C[i])
        phi.append(phi_angle)
    psi.append(0)
    return np.array(phi), np.array(psi)


def set_features1D(pdict, aa_features):
    """Run set_features1D method."""
    # code.
    """
    set_features1D:
    Args:
        pdict : pdict
    Returns:
    """
    phi, psi = compute_psi_phi(pdict)
    pdict['phi'] = phi.astype(np.float32)
    pdict['psi'] = psi.astype(np.float32)


def init_pose(po, aa_features):
    """Run init_pose method."""
    # code.
    pdict = {}
    pdict['N'], pdict['Ca'], pdict['C'], pdict['Cb'] = get_coords(po)
    pdict['nres'] = pdict['Ca'].shape[0]
    pdict['tip'] = extract_tip_coordinates(po)
    # set_lframe(pdict)
    set_neighbors6D(pdict, aa_features)    # omega6d, theta6d, phi6d
    if 'phi' in aa_features or 'psi' in aa_features:
        set_features1D(pdict, aa_features)    # phi, psi
    return pdict


def extract_sequence_from_pdb(po):
    """Run extract_sequence_from_pdb method."""
    # code.
    return [rc.restype_1to3[rc.restype_order_with_x_inv[aa]] for aa in po.aatype]


def extract_multi_distance_map(pdict):
    """Run extract_multi_distance_map method."""
    # code.
    """
    extract_multi_distance_map:
    Args:
        pdict : pdict
    Returns:
    """
    x1 = distance_matrix(pdict['Cb'], pdict['Cb'])
    x2 = distance_matrix(pdict['tip'], pdict['tip'])
    x3 = distance_matrix(pdict['Ca'], pdict['tip'])
    x4 = distance_matrix(pdict['tip'], pdict['Ca'])
    output = np.stack([x1, x2, x3, x4], axis=-1)
    return output


def extract_AAs_properties_ver1(aas, aa_features):
    """Run extract_AAs_properties_ver1 method."""
    # code.
    """
    extract_AAs_properties_ver1:
    Args:
        aas : aas
    Returns:
    """
    N_res = len(aas)
    _prop = []
    if 'aas20' in aa_features:
        aas20 = [rc.residuemap[aa] for aa in aas]
        n_values = np.max(aas20) + 1
        aas20 = np.eye(n_values)[aas20].T    # [20, N_res], residue id as one-hot vector
        _prop.append(aas20)
    if 'blosum62' in aa_features:
        blosum62 = np.stack([rc.blosummap[rc.aanamemap[aa]] for aa in aas], axis=1)    # [24, N_res], blosum62 value as one-hot vector
        _prop.append(blosum62)
    if 'rel_pos' in aa_features:
        rel_pos = np.array([min(i, N_res - i) * 1.0 / N_res * 2 for i in range(N_res)])[None, ...]  # [1, N_res]
        _prop.append(rel_pos)
    if 'meiler' in aa_features:
        meiler = np.stack([rc.meiler_features[aa] / 5 for aa in aas], axis=1)  # [7, N_res]
        _prop.append(meiler)
    if _prop:
        _prop = np.vstack(_prop)
        return _prop
    else:
        return None


def extract_USR(pdict):
    """Run extract_USR method."""
    # code.
    """
    extract_USR:
    Args:
        pdict : pdict
    Returns:
    """
    hang = pdict['nres']
    distance = distance_matrix(pdict['Ca'], pdict['Ca'])
    avg1 = []
    avg2 = []
    avg3 = []
    for i in range(hang):
        avg1.append(np.average(distance[i]))
        idx2 = np.argmax(distance, axis=1)
        avg2.append(np.average(distance[idx2[i]]))
        idx3 = np.argmax(distance[idx2[i]], axis=0)
        avg3.append(np.average(distance[idx3]))
    usr = np.vstack((avg1, avg2, avg3))
    return usr


def process(po, aa_features):
    """Run process method."""
    # code.
    pdict = init_pose(po, aa_features)    # omega6d, theta6d, phi6d, phi, psi

    if 'maps' in aa_features:
        maps = extract_multi_distance_map(pdict)
        pdict['maps'] = maps.astype(np.float32)

    ass = extract_sequence_from_pdb(po)
    prop = extract_AAs_properties_ver1(ass, aa_features)
    if 'usr' in aa_features:
        usr = extract_USR(pdict)
        if prop is not None:
            prop = np.vstack((prop, usr))
        else:
            prop = usr
    prop = prop.transpose(1, 0).astype(np.float32)
    pdict['prop'] = prop
    return pdict
