"""Code."""
import numpy as np
import torch

# standard elements (sorted by aboundance) (32)
std_elements = np.array([
    'C', 'O', 'N', 'S', 'P', 'Se', 'Mg', 'Cl', 'Zn', 'Fe', 'Ca', 'Na',
    'F', 'Mn', 'I', 'K', 'Br', 'Cu', 'Cd', 'Ni', 'Co', 'Sr', 'Hg', 'W',
    'As', 'B', 'Mo', 'Ba', 'Pt'
])    # 29

# standard residue names: AA/RNA/DNA (sorted by aboundance) (29)
std_resnames = np.array([
    'LEU', 'GLU', 'ARG', 'LYS', 'VAL', 'ILE', 'PHE', 'ASP', 'TYR',
    'ALA', 'THR', 'SER', 'GLN', 'ASN', 'PRO', 'GLY', 'HIS', 'TRP',
    'MET', 'CYS', 'G', 'A', 'C', 'U', 'DG', 'DA', 'DT', 'DC'
])    # 28

# standard atom names contained in standard residues (sorted by aboundance) (63)
std_names = np.array([
    'CA', 'N', 'C', 'O', 'CB', 'CG', 'CD2', 'CD1', 'CG1', 'CG2', 'CD',
    'OE1', 'OE2', 'OG', 'OG1', 'OD1', 'OD2', 'CE', 'NZ', 'NE', 'CZ',
    'NH2', 'NH1', 'ND2', 'CE2', 'CE1', 'NE2', 'OH', 'ND1', 'SD', 'SG',
    'NE1', 'CE3', 'CZ3', 'CZ2', 'CH2', 'P', "C3'", "C4'", "O3'", "C5'",
    "O5'", "O4'", "C1'", "C2'", "O2'", 'OP1', 'OP2', 'N9', 'N2', 'O6',
    'N7', 'C8', 'N1', 'N3', 'C2', 'C4', 'C6', 'C5', 'N6', 'N4', 'O2',
    'O4'
])    # 63


def onehot(x, v):
    """Run onehot method."""
    # code.
    m = (x.reshape(-1, 1) == np.array(v).reshape(1, -1))
    return np.concatenate([m, ~np.any(m, axis=1).reshape(-1, 1)], axis=1)


def encode_structure(structure):
    """Run encode_structure method."""
    # code.
    if isinstance(structure['xyz'], torch.Tensor):
        X = structure['xyz']
    else:
        X = torch.from_numpy(structure['xyz'].astype(np.float32))

    if isinstance(structure['resid'], torch.Tensor):
        resids = structure['resid']
    else:
        resids = torch.from_numpy(structure['resid'])
    M = (resids.unsqueeze(1) == torch.unique(resids).unsqueeze(0))

    return X, M


def encode_features(structure, device=torch.device("cpu")):
    """Run encode_features method."""
    # code.
    qe = torch.from_numpy(onehot(structure['element'], std_elements).astype(np.float32)).to(device)
    qr = torch.from_numpy(onehot(structure['resname'], std_resnames).astype(np.float32)).to(device)
    qn = torch.from_numpy(onehot(structure['name'], std_names).astype(np.float32)).to(device)

    return qe, qr, qn


def extract_topology(X, num_nn):
    """Run extract_topology method."""
    # code.
    R = X.unsqueeze(0) - X.unsqueeze(1)
    D = torch.norm(R, dim=2)    # pair distance
    D = D + torch.max(D) * (D < 1e-2).float()    # set main diagonal 0s to max d

    R = R / D.unsqueeze(2)

    knn = min(num_nn, D.shape[0])
    D_topk, ids_topk = torch.topk(D, knn, dim=1, largest=False)    # top values and top indices
    R_topk = torch.gather(R, 1, ids_topk.unsqueeze(2).repeat((1, 1, X.shape[1])))

    return ids_topk, D_topk, R_topk, D, R


def locate_contacts(xyz_i, xyz_j, r_thr, device=torch.device("cpu")):
    """Run locate_contacts method."""
    # code.
    with torch.no_grad():
        if isinstance(xyz_i, torch.Tensor):
            X_i = xyz_i.to(device)
            X_j = xyz_j.to(device)
        else:
            X_i = torch.from_numpy(xyz_i).to(device)
            X_j = torch.from_numpy(xyz_j).to(device)
        D = torch.norm(X_i.unsqueeze(1) - X_j.unsqueeze(0), dim=2)
        ids_i, ids_j = torch.where(D < r_thr)
        d_ij = D[ids_i, ids_j]

    return ids_i.cpu(), ids_j.cpu(), d_ij.cpu()


def extract_all_contacts(subunits, r_thr, device=torch.device("cpu")):
    """Run extract_all_contacts method."""
    # code.
    snames = list(subunits)
    contacts_dict = {}
    for i in range(len(snames)):
        cid_i = snames[i]

        for j in range(i + 1, len(snames)):
            cid_j = snames[j]
            ids_i, ids_j, d_ij = locate_contacts(subunits[cid_i]['xyz'], subunits[cid_j]['xyz'], r_thr, device=device)

            # insert contacts
            if (ids_i.shape[0] > 0) and (ids_j.shape[0] > 0):
                if f'{cid_i}' in contacts_dict:
                    contacts_dict[f'{cid_i}'].update({f'{cid_j}': {'ids': torch.stack([ids_i, ids_j], dim=1), 'd': d_ij}})
                else:
                    contacts_dict[f'{cid_i}'] = {f'{cid_j}': {'ids': torch.stack([ids_i, ids_j], dim=1), 'd': d_ij}}

                if f'{cid_j}' in contacts_dict:
                    contacts_dict[f'{cid_j}'].update({f'{cid_i}': {'ids': torch.stack([ids_j, ids_i], dim=1), 'd': d_ij}})
                else:
                    contacts_dict[f'{cid_j}'] = {f'{cid_i}': {'ids': torch.stack([ids_j, ids_i], dim=1), 'd': d_ij}}

    return contacts_dict
