import numpy as np
import pandas as pd
from utils.logger import Logger
from utils.constants.atom_constants import *
from utils.constants.residue_constants import *
import os
logger = Logger.logger
import torch
from np.protein import from_pdb_string
from torch_geometric.data import Data, Batch

  
def prepare_features(gt_pdb_filepath, af_pdb_filepath, chain, full_seq_AMR):
    """Run prepare_features method.
    gt_pdb_filepath: a file path of gt_pdb_filepath
    af_pdb_filepath: a file path of af_pdb_filepath
    chain: the chain of pdb

    output: a torch_geometric data of the pdb
    """
    
    # code.
    with open(af_pdb_filepath, 'r') as f:
        pdb_str = f.read()
    
    with open(gt_pdb_filepath, 'r') as f:
        gt_pdb_str = f.read()
    po = from_pdb_string(pdb_str=pdb_str, gt_pdb_str=gt_pdb_str, chain_id=chain, ca_only=False, full_seq_AMR=full_seq_AMR)
    if po == False:
        return False

    atom_mask = po.atom_mask.reshape(-1).astype(bool)

    po2struc = {
        'atom_positions': 'xyz',
        'atom_names': 'name',
        # 'atom2cgids': 'atom2cgid',
        'elements': 'element',
        'residue_names': 'resname',
        'resids': 'resid',
        'hetfields': 'het_flag',
        'icodes': 'icode'
    }
    structure = {}
    for k in [
        'atom_positions',
        'atom_names',
        'residue_names',
        # 'atom2cgids',
        'elements',
        'hetfields',
        'resids',
        'icodes',
    ]:
        item = eval(f"po.{k}")
        if k == 'atom2cgids':
            item = item.reshape(-1, item.shape[-1])
        elif k == 'atom_positions':
            item = item.reshape(-1, item.shape[-1])
        else:
            item = item.reshape(-1, )
        item = item[atom_mask] # mask the 0
        structure[po2struc[k]] = item


    X = torch.from_numpy(structure['xyz'].astype(np.float32))

    # atom2cgids = torch.from_numpy(structure['atom2cgid'].astype(np.float32))

    atom_names = structure['name'].tolist()
    atom_numbers = torch.Tensor([atom2id[atname] for atname in atom_names])     # atom number of atoms in residues. max number is 37.

    elements = structure['element'].tolist()
    atomic_numbers = [element2atomic_numbers[element]for element in elements]
    atomic_numbers = torch.Tensor(atomic_numbers)

    res_names = structure['resname'].tolist()
    res1 = [restype_3to1[resname] for resname in res_names]
    resid = [res_type12id[res1[i]] for i in range(len(res1))]
    resid = torch.Tensor(resid)

    n_nodes = X.size(0)

    # extend element 
    gt_atom_mask = torch.from_numpy(eval(f"po.{'gt_atom_mask'}").astype(np.float32))
    atom_mask = torch.from_numpy(atom_mask)

    

    atom2cgids = torch.from_numpy(eval(f"po.{'atom2cgids'}").astype(np.float32))
    gt_atom_positions = torch.from_numpy(eval(f"po.{'gt_atom_positions'}").astype(np.float32))
    all_atom_positions = torch.from_numpy(eval(f"po.{'atom_positions'}").astype(np.float32))
    gt_res_mask = torch.from_numpy(eval(f"po.{'gt_res_mask'}"))
    rmsd = torch.from_numpy(eval(f"po.{'rmsd'}").astype(np.float32))

    

    if all_atom_positions.shape[0] != gt_res_mask.shape[0]:
        return False

    trans_target = torch.from_numpy(eval(f"po.{'trans_target'}").astype(np.float32))
    rot_m4_target = torch.from_numpy(eval(f"po.{'rot_m4_target'}").astype(np.float32))

    pdb_name = af_pdb_filepath.split('/')[-1]

    data = Data(
        pos=X, 
        atom_numbers=atom_numbers, 
        atomic_numbers=atomic_numbers, 
        atom_mask=atom_mask,
        resid=resid, 
        n_nodes=n_nodes,
        gt_atom_positions=gt_atom_positions,
        gt_atom_mask=gt_atom_mask,
        gt_res_mask=gt_res_mask,
        all_atom_positions=all_atom_positions,
        atom2cgids= atom2cgids,
        pdb_name = pdb_name,
        trans_target= trans_target,
        rot_m4_target = rot_m4_target,
        rmsd = rmsd)
    return data

def load_data(csv, chain_type=None, filter=None):
    df = pd.read_csv(csv)
    logger.info(f"rows: {len(df)}")
    logger.info(f"filter={filter}")
    if filter == "ca_only":
        for pdb in ab_filter.ca_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "backbone_only":
        for pdb in ab_filter.backbone_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "long":
        for pdb in ab_filter.long_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "missing_res":
        for pdb in ab_filter.missing_res_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "missing_atom":
        for pdb in ab_filter.missing_atom_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "corner_case":
        for pdb in ab_filter.corner_case_list:
            df = df.drop(df[df["pdb"] == pdb].index) 
    elif filter == "region_le2":
        for pdb in ab_filter.region_le2_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "many_x":
        for pdb in ab_filter.many_x_res_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "all":
        for pdb in ab_filter.ca_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
        for case in ab_filter.bad_ab_list:
            pdb, chain_id = case.split("-")
            df = df.drop(df[(df["pdb"] == pdb) & (df["chain"] == chain_id)].index)
    else:
        pass
    logger.info(f"rows after drop bad pdbs: {len(df)}")
    if chain_type == "h":
        df = df[df["chain_type"] == chain_type]
        logger.info(f"rows for vh : {len(df)}")
    elif chain_type == "l":
        df = df[(df["chain_type"] == "l") | (df["chain_type"] == "k")]
        logger.info(f"rows for vl : {len(df)}")
    else:
        pass

    return df

def get_dataset_filepath(csv, sample_num = -1):
    if sample_num == -1:
        df = load_data(csv)
    else:
        df = load_data(csv).head(sample_num)
  
    # pdb_ids = [list(df["pdb"]) for df in df_list]
    chain_ids = list(df["chain"])
    pdb_files = list(df["pdb_fpath"])
    pdb_files_gt = list(df["pdb_fpath_gt"]) 
    fv_seq_amr = list(df["full_seq_AMR"]) 
    
    data_list = []
    total_data = 0
    for af_pdb, gt_pdb, chain, gt_seq in zip(pdb_files,pdb_files_gt,chain_ids,fv_seq_amr):
        if pd.isna(chain):
            chain = None
        if os.path.exists(af_pdb) and os.path.exists(gt_pdb):
            tem =prepare_features(af_pdb_filepath=af_pdb, gt_pdb_filepath=gt_pdb, chain=chain, full_seq_AMR=gt_seq) 
            if tem:
                total_data += 1
                print('total data:', total_data)
                data_list.append(tem)
            else:
                continue
        else:
            continue
        
    # af_data = prepare_features(af_pdb_filepath=af_pdb, gt_pdb_filepath=gt_pdb, chain=chain, full_seq_AMR=gt_seq) # chain id needed
    print('get data:', len(data_list))
    return data_list
    
trainfp =  '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/NB_train_dataset.csv'
train_dataset = get_dataset_filepath(trainfp, 10000)