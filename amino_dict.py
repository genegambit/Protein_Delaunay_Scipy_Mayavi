
# REQUIRED DICTIONARY.
one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
'GLY':'G', 'PRO':'P', 'CYS':'C'}

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

# List of one letter amino acid codes.
amino = ['V','I','L','E','Q','D','N','H','W','F','Y','R','K','S','T','M','A','G','P','C']


# AEI Reduced Amino Acid Representation used by Wang and Wang.
AEI = {
'C': 'i', 'M': 'i',
'F':'i', 'I':'i',
'L':'i', 'V':'i', 
'W':'i', 'Y':'i',
'A':'a', 'T':'a', 
'H':'a', 'G':'a',
'P':'a', 'R':'a', 
'D':'e', 'E':'e',
'S':'e', 'N':'e', 
'Q':'e', 'K':'e'
}

# EIS Reduced Amino Acid Representation used by Li et. al.
EIS = {
'C':'i', 'F':'i',
'Y':'i', 'W':'i',
'M':'i', 'L':'i',
'I':'i', 'V':'i',
'G':'s', 'P':'s',
'A':'s', 'T':'s',
'S':'s', 'N':'e',
'H':'e', 'Q':'e',
'E':'e', 'D':'e',
'R':'e', 'K':'e'
}

# FLP Reduced Amino Acid Representation used by Vaisman group.
FLP = {
'V':'f', 'A':'f',
'F':'f', 'I':'f',
'L':'f', 'P':'f',
'M':'f', 'G':'f',
'D':'l', 'E':'l',
'K':'l', 'R':'l',
'S':'p', 'T':'p',
'Y':'p', 'C':'p',
'N':'p', 'Q':'p',
'H':'p', 'W':'p'
}