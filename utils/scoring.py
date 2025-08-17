"""Code."""
import os

import numpy as np
import pandas as pd

from Bio.SVDSuperimposer import SVDSuperimposer

from utils.logger import Logger

logger = Logger.logger


def superimpose_np(reference, coords):
    """Run superimpose_np method."""
    # code.
    sup = SVDSuperimposer()
    sup.set(
        reference, coords
    )  # set(x, y): set the coords y will be rotated and translated on x
    sup.run()
    rot, tran = sup.get_rotran()
    return rot, tran, sup.get_transformed(), sup.get_rms()  # y_on_x, rmsd


def get_corr(preds, gts, pdbs, global_step, idx, save, mode='valid'):
    """Run get_corr method."""
    # code.
    df = pd.DataFrame()
    df['pred'] = preds
    for k in gts[0]:
        df[k] = [x[k] for x in gts]

    df['pdb'] = pdbs
    sel_labels = {}
    if "ranked_unrelax" in pdbs[0]:
        pdb_ids = list(set([x.split('/')[-2] for x in pdbs]))
    elif "structure-ensemble" in pdbs[0]:
        pdb_ids = list(set([x.split('/')[-1] for x in pdbs]))
    elif "sample100" in pdbs[0]:
        pdb_ids = list(set([x.split('/')[-1][:6] for x in pdbs]))
    else:
        pdb_ids = list(set([x.split('/')[-2][-6:] for x in pdbs]))

    spearman_rs, pearson_rs = {}, {}
    for pdb_id in pdb_ids:
        matched = df[df['pdb'].apply(lambda x: pdb_id in x)]
        sel_ix = matched['pred'].argmax()

        for k in gts[0]:
            sel_label = matched[k].values[sel_ix]
            if k not in sel_labels:
                sel_labels[k] = []
            sel_labels[k].append(sel_label)

            if k not in spearman_rs:
                spearman_rs[k] = []
            if k not in pearson_rs:
                pearson_rs[k] = []
            spearman_rs[k].append(matched[['pred', k]].corr(method='spearman').values[0, 1])
            pearson_rs[k].append(matched[['pred', k]].corr().values[0, 1])

    # logger.info(df.shape)
    # logger.info(len(sel_labels[k]))

    try:
        if not os.path.exists(os.path.join(save, f"{mode}")):
            os.mkdir(os.path.join(save, f"{mode}"))
        df.to_csv(os.path.join(save, f"{mode}", f'{mode}_set_{idx}_mode_selection_{global_step}.csv'), index=False)
    except:
        raise RuntimeError("Not found save path.")
        pass

    metrics = {}
    for k in gts[0]:
        metrics[f"pearson_r_{k}"] = df[['pred', k]].corr().values[0, 1]
        metrics[f"spearman_r_{k}"] = df[['pred', k]].corr(method='spearman').values[0, 1]
        metrics[k] = np.mean(sel_labels[k])
        metrics[f"pearson_r_per_ag_{k}"] = np.mean(pearson_rs[k])
        metrics[f"spearman_r_per_ag_{k}"] = np.mean(spearman_rs[k])

    return metrics
