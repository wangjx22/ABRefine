"""Code."""
from dataclasses import dataclass
from itertools import chain

from anarci import anarci
import numpy as np

import torch

from utils.tool_metrics.utils.logger import Logger

logger = Logger.logger

IMGT = [
    range(1, 27),
    range(27, 39),
    range(39, 56),
    range(56, 66),
    range(66, 105),
    range(105, 118),
    range(118, 129),
]

KABAT_H = [
    range(1, 31),
    range(31, 36),
    range(36, 50),
    range(50, 66),
    range(66, 95),
    range(95, 103),
    range(103, 114),
]

KABAT_L = [
    range(1, 24),
    range(24, 35),
    range(35, 50),
    range(50, 57),
    range(57, 89),
    range(89, 98),
    range(98, 114),
]


@dataclass
class Regions:
    """Define Class  Regions:."""

    fr1: str
    fr2: str
    fr3: str
    fr4: str
    cdr1: str
    cdr2: str
    cdr3: str
    seq: str


class Numbering:
    """Define Class  Numbering:."""

    def __init__(self, seq: str, scheme: str = "imgt", ncpu: int = 1):
        """Run  __init__ method."""
        # code.
        self.seq = seq
        self.scheme = scheme
        seqs = [("0", seq)]
        self.numbering, self.alignment_details, self.hit_tables = anarci(
            seqs, scheme=scheme, output=False, ncpu=ncpu
        )
        # if len(self.numbering[0]) > 1:
        #     logger.info(f"[WARNING] There are {len(self.numbering[0])} domains in {seq}")
        self.accum_L = None
        self.region = self.format_region()
        self.fegion2idx = self.find_region_idx()

    @property
    def fr1(self):
        """Run  fr1 method."""
        # code.
        return self.region.fr1

    @property
    def fr2(self):
        """Run  fr2 method."""
        # code.
        return self.region.fr2

    @property
    def fr3(self):
        """Run  fr3 method."""
        # code.
        return self.region.fr3

    @property
    def fr4(self):
        """Run  fr4 method."""
        # code.
        return self.region.fr4

    @property
    def cdr1(self):
        """Run  cdr1 method."""
        # code.
        return self.region.cdr1

    @property
    def cdr2(self):
        """Run  cdr2 method."""
        # code.
        return self.region.cdr2

    @property
    def cdr3(self):
        """Run  cdr3 method."""
        # code.
        return self.region.cdr3

    def find_region_idx(self):
        """Run  find_region_idx method."""
        # code.
        fegion2idx = {}
        buffer_len = self.seq.find(self.region.seq)
        start_idx = 0
        for name in ["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]:
            cur_region_seq = getattr(self.region, name)
            end_idx = start_idx + len(cur_region_seq)
            if name == "fr1":
                end_idx = end_idx + buffer_len
            elif name == "fr4":
                end_idx = len(self.seq)
            fegion2idx[name] = (start_idx, end_idx)
            start_idx = end_idx
        return fegion2idx

    @property
    def fr1_idx(self):
        """Run  fr1_idx method."""
        # code.
        return self.fegion2idx["fr1"]
    @property
    def fr2_idx(self):
        """Run  fr2_idx method."""
        # code.
        return self.fegion2idx["fr2"]

    @property
    def fr3_idx(self):
        """Run  fr3_idx method."""
        # code.
        return self.fegion2idx["fr3"]

    @property
    def fr4_idx(self):
        """Run  fr4_idx method."""
        # code.
        return self.fegion2idx["fr4"]

    @property
    def cdr1_idx(self):
        """Run  cdr1_idx method."""
        # code.
        return self.fegion2idx["cdr1"]

    @property
    def cdr2_idx(self):
        """Run  cdr2_idx method."""
        # code.
        return self.fegion2idx["cdr2"]

    @property
    def cdr3_idx(self):
        """Run  cdr3_idx method."""
        # code.
        return self.fegion2idx["cdr3"]

    @property
    def numbering_seq(self):
        """Run  numbering_seq method."""
        # code.
        return self.region.seq

    @property
    def is_antibody(self):
        """Run  is_antibody method."""
        # code.
        if self.numbering[0] is None:
            return False
        # if self.numbering[0] is not None and len(self.numbering[0]) > 1:
        #     logger.warning("There are %d domains in %s" % (len(self.numbering[0]), self.seq))
        else:
            return True

    def is_scfv(self):
        """Run  is_scfv method."""
        # code.
        try:
            if len(self.numbering[0]) > 1:
                return True
            else:
                return False
        except:
            logger.error(f"{self.seq} sc fv error")
            return False

    @property
    def chain_type(self):
        """Run  chain_type method."""
        # code.
        if self.numbering[0] is None or self.alignment_details[0][0]["chain_type"] is None:
            return None
        else:
            return self.alignment_details[0][0]["chain_type"].lower()

    @property
    def species(self):
        """Run  species method."""
        # code.
        return self.alignment_details[0][0]["species"].lower()

    @property
    def numbering_integer(self):
        """Run  numbering_integer method."""
        # code.
        fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list = self.get_regions()

        def _get_int(integer_str):
            """Run  _get_int method."""
            # code.
            numbers = []
            for char in integer_str:
                if char.isalpha():
                    break
                numbers.append(char)
            return int("".join(numbers))

        int_fr1 = [_get_int(item[0]) for item in fr1 if item[1] != "-"] if fr1 else []
        int_fr2 = [_get_int(item[0]) for item in fr2 if item[1] != "-"] if fr2 else []
        int_fr3 = [_get_int(item[0]) for item in fr3 if item[1] != "-"] if fr3 else []
        int_fr4 = [_get_int(item[0]) for item in fr4 if item[1] != "-"] if fr4 else []
        int_cdr1 = (
            [_get_int(item[0]) for item in cdr1 if item[1] != "-"] if cdr1 else []
        )
        int_cdr2 = (
            [_get_int(item[0]) for item in cdr2 if item[1] != "-"] if cdr2 else []
        )
        int_cdr3 = (
            [_get_int(item[0]) for item in cdr3 if item[1] != "-"] if cdr3 else []
        )

        return np.array(
            list(
                chain(int_fr1, int_cdr1, int_fr2, int_cdr2, int_fr3, int_cdr3, int_fr4)
            )
        )

    def format_region(self):
        """Run  format_region method."""
        # code.
        fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list = self.get_regions()

        str_fr1 = "".join([item[1] for item in fr1 if item[1] != "-"] if fr1 else [])
        str_fr2 = "".join([item[1] for item in fr2 if item[1] != "-"] if fr2 else [])
        str_fr3 = "".join([item[1] for item in fr3 if item[1] != "-"] if fr3 else [])
        str_fr4 = "".join([item[1] for item in fr4 if item[1] != "-"] if fr4 else [])
        str_cdr1 = "".join([item[1] for item in cdr1 if item[1] != "-"] if cdr1 else [])
        str_cdr2 = "".join([item[1] for item in cdr2 if item[1] != "-"] if cdr2 else [])
        str_cdr3 = "".join([item[1] for item in cdr3 if item[1] != "-"] if cdr3 else [])

        if self.accum_L is None:
            self.accum_L = [0]
        for reg in [str_fr1, str_cdr1, str_fr2, str_cdr2, str_fr3, str_cdr3, str_fr4]:
            self.accum_L.append(self.accum_L[-1] + len(reg))

        overall_seq = "".join(
            [str_fr1, str_cdr1, str_fr2, str_cdr2, str_fr3, str_cdr3, str_fr4]
        )
        return Regions(
            fr1=str_fr1,
            fr2=str_fr2,
            fr3=str_fr3,
            fr4=str_fr4,
            cdr1=str_cdr1,
            cdr2=str_cdr2,
            cdr3=str_cdr3,
            seq=overall_seq,
        )

    def get_regions(self):
        """Run  get_regions method."""
        # code.
        if self.scheme == "imgt":
            rng = IMGT
        elif self.scheme == "kabat" and self.chain_type == "h":
            rng = KABAT_H
        elif self.scheme == "kabat" and self.chain_type in {"k", "l"}:
            rng = KABAT_L
        else:
            raise NotImplementedError

        fr1, fr2, fr3, fr4 = [], [], [], []
        cdr1, cdr2, cdr3 = [], [], []
        type_list = []
        try:
            for item in self.numbering[0][0][0]:
                (idx, key), aa = item
                sidx = "%d%s" % (idx, key.strip())  # str index

                # # make sure numbering is in the range of IMGT
                # if len(type_list) >= rng[6][-1]:
                #     continue

                if idx in rng[0]:  # fr1
                    fr1.append([sidx, aa])
                    type_list.append("fr1")
                elif idx in rng[1]:  # cdr1
                    cdr1.append([sidx, aa])
                    type_list.append("cdr1")
                elif idx in rng[2]:  # fr2
                    fr2.append([sidx, aa])
                    type_list.append("fr2")
                elif idx in rng[3]:  # cdr2
                    cdr2.append([sidx, aa])
                    type_list.append("cdr2")
                elif idx in rng[4]:  # fr3
                    fr3.append([sidx, aa])
                    type_list.append("fr3")
                elif idx in rng[5]:  # cdr3
                    type_list.append("cdr3")
                    cdr3.append([sidx, aa])
                elif idx in rng[6]:  # fr4
                    fr4.append([sidx, aa])
                    type_list.append("fr4")
                else:
                    logger.info(f"[WARNING] seq={self.seq}, sidx={sidx}, aa={aa}")

            return fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list
        except:
            return [], [], [], [], [], [], [], []


def get_regein_feature(seq: str, numbering: Numbering):
    """Run  get_regein_feature method."""
    # code.
    """
        To annotate antibody chain sequence using anarci api

    Args:
        seq (str): amino acid sequence to annotate
        numbering: numbering object

    Reference:
        https://github.com/oxpig/ANARCI
    """
    cdr_feature = np.zeros((len(seq)))
    status = 1
    try:

        if numbering.chain_type is None:
            status = 0
            return status, cdr_feature

        buffer_len = seq.find(numbering.numbering_seq)
        start_idx = 0
        if buffer_len > 0:
            logger.info("buffer_len is not 0")
        for i, name in enumerate(["fr1", "cdr1", "fr2", "cdr2", "fr3", "cdr3", "fr4"]):
            cur_region_seq = getattr(numbering, name)
            end_idx = start_idx + len(cur_region_seq)
            if name == "fr1":
                end_idx = end_idx + buffer_len
            elif name == "fr4":
                end_idx = len(seq)
            cdr_feature[start_idx:end_idx] = i + 1
            start_idx = end_idx

    except Exception as e:
        logger.info(f"error: {seq}")
        logger.info(e)
        status = -1
        return status, cdr_feature

    return status, cdr_feature
