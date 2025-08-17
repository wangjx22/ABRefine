"""Code."""
import hashlib

import gemmi
import numpy as np

from np.residue_constants import restype_3to1
from utils.logger import Logger

logger = Logger.logger


def seq_encoder(sequence, method="md5"):
    """Run seq_encoder method."""
    # code.
    hasher = eval(f"hashlib.{method}")
    return hasher(sequence.encode(encoding="utf-8")).hexdigest()


def read_pdb(pdb_filepath):
    """Run read_pdb method."""
    # code.
    doc = gemmi.read_pdb(pdb_filepath, max_line_length=80)

    altloc_l = []
    icodes = []

    atom_element = []
    atom_name = []
    atom_xyz = []
    residue_name = []
    seq_id = []
    het_flag = []
    chain_name = []
    seq = []
    for mid, model in enumerate(doc):
        for a in model.all():
            if a.atom.has_altloc():
                key = f"{a.chain.name}_{a.residue.seqid.num}_{a.atom.name}"
                if key in altloc_l:
                    continue
                else:
                    altloc_l.append(key)

            icodes.append(a.residue.seqid.icode.strip())

            atom_element.append(a.atom.element.name)
            atom_name.append(a.atom.name)
            atom_xyz.append([a.atom.pos.x, a.atom.pos.y, a.atom.pos.z])
            residue_name.append(a.residue.name)
            seq_id.append(a.residue.seqid.num)
            het_flag.append(a.residue.het_flag)
            chain_name.append(f"{a.chain.name}:{mid}")
            seq.append(restype_3to1[a.residue.name])

    return {
        'xyz': np.array(atom_xyz, dtype=np.float32),
        'name': np.array(atom_name),
        'element': np.array(atom_element),
        'resname': np.array(residue_name),
        'resid': np.array(seq_id, dtype=np.int32),
        'het_flag': np.array(het_flag),
        'chain_name': np.array(chain_name),
        'icode': np.array(icodes),
        'seq': np.array(seq),
    }


def clean_structure(structure, rm_wat=True):
    """Run clean_structure method."""
    # code.
    m_wat = (structure["resname"] == "HOH")
    m_h = (structure["element"] == "H")
    m_d = (structure["element"] == "D")
    m_hwat = (structure["resname"] == "DOD")

    if rm_wat:

        mask = ((~m_wat) & (~m_h) & (~m_d) & (~m_hwat))
    else:
        mask = ((~m_h) & (~m_d) & (~m_hwat))
        structure["resid"][m_wat] = -999

    structure = {key: structure[key][mask] for key in structure}

    chains = structure["chain_name"]
    ids_chains = np.where(np.array(chains).reshape(-1, 1) == np.unique(chains).reshape(1, -1))[1]
    delta_chains = np.abs(np.sign(np.concatenate([[0], np.diff(ids_chains)])))
    icodes = structure["icode"]
    ids_icodes = np.where(np.array(icodes).reshape(-1, 1) == np.unique(icodes).reshape(1, -1))[1]
    delta_icodes = np.abs(np.sign(np.concatenate([[0], np.diff(ids_icodes)])))

    resids = structure["resid"]
    delta_resids = np.abs(np.sign(np.concatenate([[0], np.diff(resids)])))

    resids = np.cumsum(np.sign(delta_chains + delta_resids + delta_icodes)) + 1

    structure['resid'] = resids

    structure.pop("icode")

    return structure


def atom_select(structure, sel):
    """Run atom_select method."""
    # code.
    return {key: structure[key][sel] for key in structure}


def split_by_chain(structure):
    """Run split_by_chain method."""
    # code.
    chains = {}
    cnames = structure["chain_name"]
    ucnames = np.unique(cnames)
    m_chains = (cnames.reshape(-1, 1) == np.unique(cnames).reshape(1, -1))

    for i in range(len(ucnames)):
        chain = atom_select(structure, m_chains[:, i])
        chain.pop("chain_name")
        chains[ucnames[i]] = chain

    return chains


def concatenate_chains(chains):
    """Run concatenate_chains method."""
    # code.
    keys = set.intersection(*[set(chains[cid]) for cid in chains])
    structure = {key: np.concatenate([chains[cid][key] for cid in chains]) for key in keys}

    structure['chain_name'] = np.concatenate([np.array([cid] * chains[cid]['xyz'].shape[0]) for cid in chains])    # [N_atom]

    return structure


def tag_hetatm_chains(structure):
    """Run tag_hetatm_chains method."""
    # code.
    m_hetatm = (structure['het_flag'] == "H")
    resids_hetatm = structure['resid'][m_hetatm]

    delta_hetatm = np.cumsum(np.abs(np.sign(np.concatenate([[0], np.diff(resids_hetatm)]))))
    cids_hetatm = np.array([f"{cid}:{hid}" for cid, hid in zip(structure['chain_name'][m_hetatm], delta_hetatm)])
    cids = structure['chain_name'].copy().astype(np.dtype('<U10'))
    cids[m_hetatm] = cids_hetatm
    structure['chain_name'] = np.array(list(cids)).astype(str)

    return structure


def remove_duplicate_tagged_subunits(subunits):
    """Run remove_duplicate_tagged_subunits method."""
    # code.
    tagged_cids = [cid for cid in subunits if (len(cid.split(':')) == 3)]
    for i in range(len(tagged_cids)):
        cid_i = tagged_cids[i]
        for j in range(i + 1, len(tagged_cids)):
            cid_j = tagged_cids[j]

            if (cid_i in subunits) and (cid_j in subunits):
                xyz0 = subunits[cid_i]['xyz']
                xyz1 = subunits[cid_j]['xyz']

                if xyz0.shape[0] == xyz1.shape[0]:
                    d_min = np.min(np.linalg.norm(xyz0 - xyz1, axis=1))
                    if d_min < 0.2:
                        subunits.pop(cid_j)

    return subunits


def filter_non_atomic_subunits(subunits):
    """Run filter_non_atomic_subunits method."""
    # code.
    for sname in list(subunits):
        n_res = np.unique(subunits[sname]['resid']).shape[0]
        n_atm = subunits[sname]['xyz'].shape[0]

        if (n_atm == n_res) & (n_atm > 1):
            subunits.pop(sname)

    return subunits


def encode_bfactor(structure, p):
    """Run encode_bfactor method."""
    # code.
    names = structure["name"]
    elements = structure["element"]
    het_flags = structure["het_flag"]
    m_ca = (names == "CA") & (elements == "C") & (het_flags == "A")
    resids = structure["resid"]

    if p.shape[0] == m_ca.shape[0]:
        structure['bfactor'] = p

    elif p.shape[0] == np.sum(m_ca):

        bf = np.zeros(len(resids), dtype=np.float32)
        for i in np.unique(resids):
            m_ri = (resids == i)
            i_rca = np.where(m_ri[m_ca])[0]
            if len(i_rca) > 0:
                bf[m_ri] = float(np.max(p[i_rca]))

        structure['bfactor'] = bf

    elif p.shape[0] == np.unique(resids).shape[0]:
        uresids = np.unique(resids)
        bf = np.zeros(len(resids), dtype=np.float32)
        for i in uresids:
            m_ri = (resids == i)
            m_uri = (uresids == i)
            bf[m_ri] = float(np.max(p[m_uri]))

        structure['bfactor'] = bf

    else:
        logger.info("WARNING: bfactor not saved")

    return structure


def get_interface(subunits, interface_cut, look_level='atom'):
    """Run get_interface method."""
    # code.
    chains = list(subunits)
    # for chain in subunits:
    #     first_look = [0]
    #     cur_id = subunits[chain]['resid'][0]
    #     for ix, i in enumerate(subunits[chain]['resid']):
    #         if i != cur_id:
    #             first_look.append(ix)
    #             cur_id = i
    #     seq = ''.join(subunits[chain]['seq'][first_look])
    #     is_ab, chain_type = is_antibody(seq)
    #     if not is_ab:
    #         chain_type = 'protein'
    #     if chain_type in ['l', 'k']:
    #         chain_type = 'l'
    #     chain_orders[chain_type] = chain
    #
    new_subunits = {
        'A:0': subunits[chains[0]],
        'B:0': subunits[chains[1]],
    }

    ab_xyz = new_subunits['A:0']['xyz']
    ag_xyz = new_subunits['B:0']['xyz']

    dists = np.sum(
        (ab_xyz[..., None, :] - ag_xyz[..., None, :, :]) ** 2,
        axis=-1,
    )
    ab_dist = dists.min(axis=-1) < interface_cut
    ag_dist = dists.min(axis=0) < interface_cut

    no_interface = []
    for chain, select in zip(new_subunits, [
        ab_dist, ag_dist
    ]):
        if select.sum():
            for k in new_subunits[chain]:
                new_subunits[chain][k] = new_subunits[chain][k][select]
        else:
            no_interface.append(chain)
    for chain in no_interface:
        new_subunits.pop(chain)
    if not new_subunits:
        raise ValueError("no interface found")

    return new_subunits
