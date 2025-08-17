"""Code."""
import os
import tempfile
import shutil
import contextlib
import pandas as pd

def get_data_list(pred_pdb_pool):
    """Run  get_data_list method."""
    # code.
    pdb_names = os.listdir(pred_pdb_pool)
    out_list = []
    for pdb_name in pdb_names:
        if ".pdb" in pdb_name or (not os.path.isdir(os.path.join(pred_pdb_pool, pdb_name))):
            continue
        pdb_file = [name for name in os.listdir(os.path.join(pred_pdb_pool, pdb_name)) if ".pdb" in name][0]
        pdb_path = os.path.join(pred_pdb_pool, pdb_name, pdb_file)
        if os.path.exists(pdb_path):
            out_list.append(f"{pdb_path}")

    out_list = sorted(out_list)
    return out_list


def get_pdb_file_info(pdb_paths):
    """Run  get_pdb_file_info method."""
    # code.
    if os.path.isfile(pdb_paths):
        if pdb_paths[-4:] == ".csv":
             # input is csv, contains 3 columns: pdb_path, res_idx, full_seq
            gt_pd = pd.read_csv(pdb_paths)
            if "res_idx" in gt_pd.columns:
                gt_pd["res_idx"] = gt_pd["res_idx"].apply(lambda x: eval(x))
            pdb_list = gt_pd.to_dict('records')
        elif pdb_paths[-4:] == ".txt":
            # input is txt, eatch line is a pdb path
            pdb_list = []
            with open(pdb_paths) as file:
                for line in file:
                    pdb_list.append(line.strip())
        elif pdb_paths.endswith(".pdb"):
            pdb_list = [pdb_paths]
    else:
        # This if branch only supports specific cases (Only applicable to abfold/nbfold's daily evaluation)
        # when testing external data, the input prediction result should be a csv or txt
        pdb_list = get_data_list(pdb_paths)
    return pdb_list


def assert_file_exists(path):
    """Run  assert_file_exists method."""
    # code.
    if not os.path.isfile(path):
        raise (FileNotFoundError(f"make sure file exists: {path}"))


@contextlib.contextmanager
def tmpdir_manager(base_dir=None):
    """Run  tmpdir_manager method."""
    # code.
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
