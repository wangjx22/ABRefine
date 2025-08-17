from anarci import anarci

import re

def find_substring_indices_regex(s, substring):
    match = re.search(re.escape(substring), s)
    if match is None:
        return None 
    return match.start(), match.end() - 1


aa_codes = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "LYS": "K",
    "ILE": "I",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TYR": "Y",
    "TRP": "W",
}

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


def regions(seq, numbering, rng):
    fr1, fr2, fr3, fr4 = [], [], [], []
    cdr1, cdr2, cdr3 = [], [], []
    type_list = []

    for item in numbering[0][0][0]:
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
            print(f"[WARNING] seq={seq}, sidx={sidx}, aa={aa}")

    return fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list


def make_numbering_by_api(
    seq, input_species=None, scheme="imgt", input_chain_type=None, ncpu=4
):
    seqs = [("0", seq)]
    numbering, alignment_details, hit_tables = anarci(
        seqs, scheme=scheme, output=False, ncpu=ncpu
    )
    if len(numbering[0]) > 1:
        print(f"[WARNING] There are {len(numbering[0])} domains in {seq}")

    species = alignment_details[0][0]["species"].lower()
    chain_type = alignment_details[0][0]["chain_type"].lower()
    e_value = alignment_details[0][0]["evalue"]
    score = alignment_details[0][0]["bitscore"]
    v_start = alignment_details[0][0]["query_start"]
    v_end = alignment_details[0][0]["query_end"]
    if scheme == "imgt":
        rng = IMGT
    elif scheme == "kabat" and chain_type.lower() == "h":
        rng = KABAT_H
    elif scheme == "kabat" and (chain_type.lower() == "l" or chain_type.lower() == "k"):
        rng = KABAT_L
    else:
        raise NotImplementedError
    fr1, fr2, fr3, fr4, cdr1, cdr2, cdr3, type_list = regions(seq, numbering, rng)

    str_fr1 = "".join([item[1] for item in fr1 if item[1] != "-"])
    str_fr2 = "".join([item[1] for item in fr2 if item[1] != "-"])
    str_fr3 = "".join([item[1] for item in fr3 if item[1] != "-"])
    str_fr4 = "".join([item[1] for item in fr4 if item[1] != "-"])
    str_cdr1 = "".join([item[1] for item in cdr1 if item[1] != "-"])
    str_cdr2 = "".join([item[1] for item in cdr2 if item[1] != "-"])
    str_cdr3 = "".join([item[1] for item in cdr3 if item[1] != "-"])
    return str_fr1, str_fr2, str_fr3, str_fr4, str_cdr1, str_cdr2, str_cdr3


def is_antibody(seq, scheme="imgt", ncpu=4):
    seqs = [("0", seq)]
    numbering, alignment_details, hit_tables = anarci(
        seqs, scheme=scheme, output=False, ncpu=ncpu
    )
    if numbering[0] is None:
        return False, None

    if numbering[0] is not None and len(numbering[0]) > 1:
        print("There are %d domains in %s" % (len(numbering[0]), seq))

    chain_type = alignment_details[0][0]["chain_type"].lower()
    if chain_type is None:
        return False, None
    else:
        if chain_type == "k":
            chain_type = "l"  # kappa is one kind of light chain
        return True, chain_type


if __name__ == "__main__":
    seq = "EVQLVESGGGLVQPGGSLRLSCAASEITVSSNYMNWVRQAPGKGLEWVSVIYPGGTTYYAESVKGRFAISRDNSKNTLYLQMNSLRPEDTAVYYCARVMRHEIWGQGTLVTVSSDIQMTQSQSSLSASVGDRVTITCQASQDINNFLNWYQQKPGKAPKLLIYDASHLETGVPSRFSGSGSGTNFTFTISSLQPEDIATYYCQHCDNPPYTFGQGTKLEIR"
    #seq = "EVRLVESGGGLVQPGGSLRLSCAASGFTFSDYYISWVRQAPGRGPEWVGFIRNVLYRGTTEYAPSVKGRFIISRDDSRAIASLQMNGLKADDTAVYYCALGASGTDRDWFDVWGPGVLVTVSS"
    seq = "QVQLVESGGGLVQPGGSLRLSCAASRSISSINIMGWYRQAPGKERESVASHTRDGSTDYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCTTLTGFPRIRSWGQGTQVTVSS"
    print(seq, len(seq))
    res = make_numbering_by_api(seq)
    print(res)
    n = 0
    for s in res:
        n += len(s)
    print(n)
    print(find_substring_indices_regex(seq, res[-1]))
    print(find_substring_indices_regex(seq, res[-2]))
    print(find_substring_indices_regex(seq, res[-3]))
