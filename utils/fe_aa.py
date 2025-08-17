"""Code."""
import json
import numpy as np
import torch
import os

aa_level_methods = ['string_vec5','vec5_CTC','One_hot', 'One_hot_6_bit', 'Binary_5_bit', 'Hydrophobicity_matrix', 'Meiler_parameters', 'Acthely_factors', 'PAM250', 'BLOSUM62', 'Miyazawa_energies', 'Micheletti_potentials', 'AESNN3', 'ANN4D']
residue_types = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','X']

env_path = '%s/../' % os.path.dirname(os.path.abspath(__file__))

aa_emd_lib=dict()
for item in aa_level_methods:
    if item.lower() in ['string_vec5','vec5_ctc']:
        aa_encoding=dict()
        with open("%s/static/%s.txt" % (env_path, item), 'r') as f:
            for line in f.readlines():
                aa,fe=line.split('\t')
                fe = [float(f1) for f1 in fe.split(' ')]
                aa_encoding[aa]=fe
    else:
        with open("%s/static/%s.json" % (env_path, item), 'r') as load_f:
            aa_encoding = json.load(load_f)
    aa_emd_lib[item.lower()]=aa_encoding

def protein_encoding_with_aa_emd(feature_names, seq):
    """Run protein_encoding_with_aa_emd method."""
    # code.
    ## aa_lvel_feature
    all_fe = []
    for item in aa_level_methods:
        if item.lower() in feature_names:
            aa_encoding = aa_emd_lib[item.lower()]
            seq = seq.upper()
            encoding_data = []
            for res in seq:
                if res not in residue_types:
                    res = "X"
                encoding_data.append(aa_encoding[res])
            assert len(encoding_data) == len(seq)
            feat = np.array(encoding_data)
            fe= torch.from_numpy(feat.astype(np.float32))
            all_fe.append(fe)

    return torch.concatenate(all_fe, dim=-1)
