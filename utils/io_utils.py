"""Code."""
import gzip


def read_pdb_str(pdb_name):
    """Run read_pdb_str method."""
    # code.
    """

    Args:
        pdb_name:
            str, pdb name
    Return:
        pdb_str:
            file content
    """
    if pdb_name.endswith(".gz"):
        with gzip.open(pdb_name, "rt", encoding="utf-8") as f:
            pdb_str = f.read()
    else:
        with open(pdb_name, "r") as f:
            pdb_str = f.read()

    return pdb_str
