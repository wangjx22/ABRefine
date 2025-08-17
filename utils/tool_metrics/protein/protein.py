"""Code."""
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein data type."""
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Union, Tuple, List
import io
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from . import residue_constants
from Bio.PDB import PDBParser
from Bio import Align
from anarci import anarci

from utils.tool_metrics.utils.logger import Logger
from utils.tool_metrics.numbering import Numbering

logger = Logger.logger
aligner = Align.PairwiseAligner()

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]  # Is a nested dict.
PICO_TO_ANGSTROM = 0.01

# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # 0-indexed number corresponding to the chain in the protein that this
    # residue belongs to
    chain_index: np.ndarray

    chain_ids: [str] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

    # protein resolution
    resolution: any = None

    def __post_init__(self):
        """Run  __post_init__ method."""
        # code.
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} "
                "chains because these cannot be written to PDB format"
            )


def filter_aa_from_seq(seq):
    """Run  filter_aa_from_seq method."""
    # code.
    aa_index = []
    for idx in range(len(seq)):
        if seq[idx] == "-":
            continue
        else:
            aa_index.append(idx)
    return aa_index


def from_pdb_string(
    pdb_str: str,
    chain_id: Optional[str] = None,
    missing_seq: Optional[str] = None,
    missing_idx: Optional[list] = None,
    is_cut_ab_fv: bool = False,
    resolution: float = None,
    is_gt=False,
    return_id2seq=False,
    skip_x=False,
) -> Union[Tuple[Protein, int, List], Tuple[Protein, int]]:
    """Run  from_pdb_string method."""
    # code.
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain is
      parsed. Else, all chains are parsed.
      is_cut_ab_fv: whether to cut ab fv
      resolution: the resolution of PDB

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    if resolution is None:
        resolution = structure.header["resolution"]
        if resolution is None:
            resolution = 0.0
    models = list(structure.get_models())
    model = models[0]
    pdb_chains = list(model.get_chains())
    chain_id_list = chain_id
    if chain_id is not None:
        if "," in chain_id:
            chain_id_list = chain_id.split(",")
        chain_model = []
        for id in chain_id_list:
            for pdb_chain in pdb_chains:
                if id == pdb_chain.id:
                    chain_model.append(pdb_chain)
        pdb_chains = chain_model

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    id2seq = {}
    keep_aa_idx = []
    last_aa_idx = 0
    status = 1
    nums = []
    for chain in pdb_chains:
        if chain_id is not None and chain.id not in chain_id_list:
            continue
        seqs = []
        cur_chain_aatype = []
        cur_atom_positions = []
        cur_atom_mask = []
        cur_residue_index = []
        cur_chain_ids = []
        cur_b_factors = []
        buffer = 0
        if missing_idx is not None and missing_seq is not None:
            assert missing_seq is not None
            assert missing_idx is not None
            residue_dict = dict()
            for res in chain:
                if res.id[0] != " ":  # skip HETATM
                    continue
                if res.id[2] != " ":
                    resid = str(res.id[1]) + res.id[2]
                else:
                    resid = str(res.id[1])
                residue_dict[resid] = res
            last_ca_coor = None
            for idx in range(len(missing_idx)):
                res_idx = missing_idx[idx]
                res = residue_dict.get(res_idx)
                res_shortname = missing_seq[idx]
                if skip_x and res_shortname == "X":
                    continue
                seqs.append(res_shortname)
                restype_idx = residue_constants.restype_order.get(
                    res_shortname, residue_constants.restype_num
                )
                pos = np.zeros((residue_constants.atom_type_num, 3))
                mask = np.zeros((residue_constants.atom_type_num,))
                res_b_factors = np.zeros((residue_constants.atom_type_num,))
                dist = None
                if res is not None:
                    assert res_shortname == residue_constants.restype_3to1.get(
                        res.resname, "X"
                    )
                    ca_exits_flag = False
                    cur_ca_coor = None
                    for atom in res:
                        if atom.name not in residue_constants.atom_types:
                            continue
                        pos[residue_constants.atom_order[atom.name]] = atom.coord
                        mask[residue_constants.atom_order[atom.name]] = 1.0
                        res_b_factors[
                            residue_constants.atom_order[atom.name]
                        ] = atom.bfactor
                        if atom.name == "CA":
                            ca_exits_flag = True
                            cur_ca_coor = np.array(atom.coord)
                    if last_ca_coor is not None and ca_exits_flag:
                        dist = np.linalg.norm(cur_ca_coor - last_ca_coor)

                    last_ca_coor = cur_ca_coor
                cur_chain_aatype.append(restype_idx)
                cur_atom_positions.append(pos)
                cur_atom_mask.append(mask)
                if res_idx[-1].isalpha():
                    buffer = buffer + 1
                    res_idx = res_idx[:-1]
                if dist is not None and dist < 4:
                    cur_residue_index.append(cur_residue_index[-1] + 1)
                else:
                    cur_residue_index.append(int(res_idx) + buffer)
                # chain.id is used for multiple chains
                cur_chain_ids.append(chain.id)
                cur_b_factors.append(res_b_factors)
        else:
            last_ca_coor = None
            dist = None
            lsat_res_id = None
            for index, res in enumerate(chain):
                if res.id[0] != " ":  # skip HETATM
                    continue
                if res.id[2] != " " and lsat_res_id == res.id[1]:
                    buffer = buffer + 1
                res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
                if skip_x and res_shortname == "X":
                    continue
                seqs.append(res_shortname)
                restype_idx = residue_constants.restype_order.get(
                    res_shortname, residue_constants.restype_num
                )
                pos = np.zeros((residue_constants.atom_type_num, 3))
                mask = np.zeros((residue_constants.atom_type_num,))
                res_b_factors = np.zeros((residue_constants.atom_type_num,))
                ca_exits_flag = False
                for atom in res:
                    if atom.name not in residue_constants.atom_types:
                        continue
                    if atom.name == "CA":
                        ca_exits_flag = True
                        cur_ca_coor = np.array(atom.coord)
                    pos[residue_constants.atom_order[atom.name]] = atom.coord
                    mask[residue_constants.atom_order[atom.name]] = 1.0
                    res_b_factors[
                        residue_constants.atom_order[atom.name]
                    ] = atom.bfactor
                if last_ca_coor is not None and ca_exits_flag:
                    dist = np.linalg.norm(cur_ca_coor - last_ca_coor)
                    # only when ca_exists_flag is true, we will update the last_ca_coor
                    last_ca_coor = cur_ca_coor

                if np.sum(mask) < 0.5:
                    # If no known atom positions are reported for the residue then skip it.
                    continue

                cur_chain_aatype.append(restype_idx)
                cur_atom_positions.append(pos)
                cur_atom_mask.append(mask)
                #
                if is_gt:
                    if dist is not None and dist < 4:
                        cur_residue_index.append(cur_residue_index[-1] + 1)
                    else:
                        cur_residue_index.append(res.id[1] + buffer)
                else:
                    cur_residue_index.append(res.id[1] + buffer)
                cur_chain_ids.append(chain.id)
                cur_b_factors.append(res_b_factors)
                lsat_res_id = res.id[1]

        aatype.extend(cur_chain_aatype)
        atom_positions.extend(cur_atom_positions)
        atom_mask.extend(cur_atom_mask)
        residue_index.extend(cur_residue_index)
        chain_ids.extend(cur_chain_ids)
        b_factors.extend(cur_b_factors)

        seqs = "".join(seqs)
        id2seq[chain.id] = cur_chain_aatype
        if is_cut_ab_fv:
            num = Numbering(seqs)
            if not num.numbering_seq:
                num = Numbering(seqs, bit_score_threshold=20)
            nums.append(num)
            flag_ab = num.is_antibody
            if flag_ab:
                numbering_seq = num.numbering_seq
                try:
                    start_idx = seqs.find(numbering_seq)
                    end_idx = start_idx + len(numbering_seq)
                    index = [i + last_aa_idx for i in range(start_idx, end_idx)]
                    id2seq[chain.id] = id2seq[chain.id][start_idx: end_idx]
                except Exception as e:
                    status = -1
                    logger.error(f"cut_ab_fv exception: {e}")
                    index = [i + last_aa_idx for i in range(len(seqs))]
            else:
                index = [i + last_aa_idx for i in range(len(seqs))]

            keep_aa_idx.extend(index)
            last_aa_idx += len(seqs)
    if is_cut_ab_fv:
        aatype = [aatype[idx] for idx in keep_aa_idx]
        atom_positions = [atom_positions[idx] for idx in keep_aa_idx]
        atom_mask = [atom_mask[idx] for idx in keep_aa_idx]
        residue_index = [residue_index[idx] for idx in keep_aa_idx]
        chain_ids = [chain_ids[idx] for idx in keep_aa_idx]
        b_factors = [b_factors[idx] for idx in keep_aa_idx]

    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    protein = Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        chain_index=chain_index,
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors),
        resolution=resolution,
    )
    if return_id2seq:
        return (protein, id2seq)
    else:
        return protein


def is_antibody(seq, scheme="imgt", ncpu=1):
    """Run  is_antibody method."""
    # code.
    seqs = [("0", seq)]
    numbering, alignment_details, hit_tables = anarci(
        seqs, scheme=scheme, output=False, ncpu=ncpu
    )
    if numbering[0] is None:
        return False, None

    if numbering[0] is not None and len(numbering[0]) > 1:
        logger.warning("There are %d domains in %s" % (len(numbering[0]), seq))

    chain_type = alignment_details[0][0]["chain_type"].lower()
    if chain_type is None:
        return False, None
    else:
        return True, chain_type
