"""Code."""
import numpy as np
from dataclasses import dataclass
from anarci import anarci

from utils.logger import Logger

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


@dataclass
class Regions:
    """Define Class Regions."""

    fr1: str
    fr2: str
    fr3: str
    fr4: str
    cdr1: str
    cdr2: str
    cdr3: str
    seq: str
    idx_list: list


class Numbering:
    """Define Class Numbering."""

    def __init__(
        self,
        seq: str,
        scheme: str = "imgt",
        ncpu: int = 4,
        bit_score_threshold: int = 80,
        seq_is_fv=False,
    ):
        """Run __init__ method."""
        # code.
        self.seq = seq
        self.seq_is_fv = seq_is_fv
        self.scheme = scheme
        seqs = [("0", seq)]
        self.numbering, self.alignment_details, self.hit_tables = anarci(
            seqs,
            scheme=scheme,
            output=False,
            ncpu=ncpu,
            bit_score_threshold=bit_score_threshold,
        )
        # if len(self.numbering[0]) > 1:
        #     logger.info(f"[WARNING] There are {len(self.numbering[0])} domains in {seq}")
        self.region = self.format_region()

    @property
    def fr1(self):
        """Run fr1 method."""
        # code.
        return self.region.fr1

    @property
    def fr2(self):
        """Run fr2 method."""
        # code.
        return self.region.fr2

    @property
    def fr3(self):
        """Run fr3 method."""
        # code.
        return self.region.fr3

    @property
    def fr4(self):
        """Run fr4 method."""
        # code.
        return self.region.fr4

    @property
    def cdr1(self):
        """Run cdr1 method."""
        # code.
        return self.region.cdr1

    @property
    def cdr2(self):
        """Run cdr2 method."""
        # code.
        return self.region.cdr2

    @property
    def cdr3(self):
        """Run cdr3 method."""
        # code.
        return self.region.cdr3

    @property
    def numbering_seq(self):
        """Run numbering_seq method."""
        # code.
        return self.region.seq

    @property
    def numbering_residue_index(self):
        """Run numbering_residue_index method."""
        # code.
        return np.array(self.region.idx_list)

    @property
    def is_antibody(self):
        """Run is_antibody method."""
        # code.
        if self.numbering[0] is None:
            return False
        # if self.numbering[0] is not None and len(self.numbering[0]) > 1:
        #     logger.warning("There are %d domains in %s" % (len(self.numbering[0]), self.seq))
        else:
            return True

    @property
    def chain_type(self):
        """Run chain_type method."""
        # code.
        return self.alignment_details[0][0]["chain_type"].lower()

    @property
    def species(self):
        """Run species method."""
        # code.
        return self.alignment_details[0][0]["species"].lower()

    def format_region(self):
        """Run format_region method."""
        # code
        (
            fr1,
            fr2,
            fr3,
            fr4,
            cdr1,
            cdr2,
            cdr3,
            type_list,
            num_idx_list,
        ) = self.get_regions()
        str_fr1 = "".join([item[1] for item in fr1 if item[1] != "-"] if fr1 else [])
        str_fr2 = "".join([item[1] for item in fr2 if item[1] != "-"] if fr2 else [])
        str_fr3 = "".join([item[1] for item in fr3 if item[1] != "-"] if fr3 else [])
        str_fr4 = "".join([item[1] for item in fr4 if item[1] != "-"] if fr4 else [])
        str_cdr1 = "".join([item[1] for item in cdr1 if item[1] != "-"] if cdr1 else [])
        str_cdr2 = "".join([item[1] for item in cdr2 if item[1] != "-"] if cdr2 else [])
        str_cdr3 = "".join([item[1] for item in cdr3 if item[1] != "-"] if cdr3 else [])
        fv_seq = "".join(
            [str_fr1, str_cdr1, str_fr2, str_cdr2, str_fr3, str_cdr3, str_fr4]
        )

        if self.seq_is_fv and fv_seq != self.seq:
            logger.info(f"seq={self.seq}")
            logger.info(f"fv_seq={fv_seq}")
            logger.info(f"num_idx_list={num_idx_list}")
            be = self.seq.find(fv_seq)
            if be > 0:
                str_fr1 = self.seq[:be] + str_fr1
                fv_seq = self.seq[:be] + fv_seq
                min_idx = min(num_idx_list)
                add_list = list()
                for i in range(be):
                    # this strategy may be not correct enough.
                    add_list.append(max(min_idx - be + i, 0))
                num_idx_list = add_list + num_idx_list

            gap = len(self.seq) - len(fv_seq)
            if gap < 0:
                logger.error(f"fv_seq={fv_seq}, seq={self.seq}")
            if gap > 0:
                str_fr4 = str_fr4 + self.seq[-gap:]
                fv_seq = fv_seq + self.seq[-gap:]
                max_idx = max(num_idx_list)
                add_list = list()
                for i in range(gap):
                    # this strategy may be not correct enough.
                    add_list.append(max_idx + i + 1)
                num_idx_list = num_idx_list + add_list

            logger.info("**********************************")
            logger.info(f"fv_seq={fv_seq}")
            logger.info(f"num_idx_list={num_idx_list}")

        if len(fv_seq) != len(num_idx_list):
            logger.error(fv_seq)

        return Regions(
            fr1=str_fr1,
            fr2=str_fr2,
            fr3=str_fr3,
            fr4=str_fr4,
            cdr1=str_cdr1,
            cdr2=str_cdr2,
            cdr3=str_cdr3,
            seq=fv_seq,
            idx_list=num_idx_list,
        )

    def get_regions(self):
        """Run get_regions method."""
        # code
        if self.scheme == "imgt":
            rng = IMGT
        else:
            raise NotImplementedError
        fr1, fr2, fr3, fr4 = [], [], [], []
        cdr1, cdr2, cdr3 = [], [], []
        type_list = []
        num_idx_list = list()
        if self.numbering[0] is None:
            return fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list

        for item in self.numbering[0][0][0]:
            (idx, key), aa = item
            sidx = "%d%s" % (idx, key.strip())  # str index
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
                continue

            if aa != "-":
                num_idx_list.append(int(idx))

        return fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list, num_idx_list


class ChothiaNumbering:
    """Define Class ChothiaNumbering."""

    def __init__(
        self,
        seq: str,
        scheme: str = "chothia",
        ncpu: int = 4,
        bit_score_threshold: int = 80,
        seq_is_fv=False,
    ):
        """Run __init__ method."""
        # code
        self.seq = seq
        self.seq_is_fv = seq_is_fv
        self.scheme = scheme
        seqs = [("0", seq)]
        self.numbering, self.alignment_details, self.hit_tables = anarci(
            seqs,
            scheme=scheme,
            output=False,
            ncpu=ncpu,
            bit_score_threshold=bit_score_threshold,
        )
        self.num_idx_list = self.get_num_idx_list()

    @property
    def numbering_residue_index(self):
        """Run numbering_residue_index method."""
        # code
        return np.array(self.num_idx_list)

    def get_num_idx_list(self):
        """Run get_num_idx_list method."""
        # code
        num_idx_list = list()

        fv_seq = list()
        for item in self.numbering[0][0][0]:
            (idx, key), aa = item
            if aa != "-":
                num_idx_list.append(int(idx))
                fv_seq.append(aa)

        # solving corner case
        fv_seq = "".join(fv_seq)
        if self.seq_is_fv and fv_seq != self.seq and len(fv_seq) > 0:
            logger.info(f"seq={self.seq}")
            logger.info(f"fv_seq={fv_seq}")
            logger.info(f"num_idx_list={num_idx_list}")
            be = self.seq.find(fv_seq)
            if be > 0:
                fv_seq = self.seq[:be] + fv_seq
                min_idx = min(num_idx_list)
                add_list = list()
                for i in range(be):
                    # this strategy may be not correct enough.
                    add_list.append(max(min_idx - be + i, 0))
                num_idx_list = add_list + num_idx_list

            gap = len(self.seq) - len(fv_seq)
            if gap < 0:
                logger.error(f"fv_seq={fv_seq}, seq={self.seq}")
            if gap > 0:
                fv_seq = fv_seq + self.seq[-gap:]
                max_idx = max(num_idx_list)
                add_list = list()
                for i in range(gap):
                    # this strategy may be not correct enough.
                    add_list.append(max_idx + i + 1)
                num_idx_list = num_idx_list + add_list

            logger.info("**********************************")
            logger.info(f"fv_seq={fv_seq}")
            logger.info(f"num_idx_list={num_idx_list}")

        return num_idx_list
