# Define the TIMIT to ARPAbet mapping
TIMIT_TO_ARPABET = {
    'aa': 'AA',    # One-to-one
    'ae': 'AE',    # One-to-one
    'ah': 'AH',    # One-to-one
    'ao': 'AO',    # One-to-one
    'aw': 'AW',    # One-to-one
    'ax': 'AH',    # Many-to-one (ax, ax-h -> AH)
    'ax-h': 'AH',  # Many-to-one (ax, ax-h -> AH)
    'axr': 'ER',   # One-to-one (axr -> ER)
    'ay': 'AY',    # One-to-one
    'b': 'B',      # One-to-one
    'bcl': 'B',    # Many-to-one (b, bcl -> B)
    'ch': 'CH',    # One-to-one
    'd': 'D',      # One-to-one
    'dcl': 'D',    # Many-to-one (d, dcl -> D)
    'dh': 'DH',    # One-to-one
    'dx': 'D',     # One-to-one (approximated to D)
    'eh': 'EH',    # One-to-one
    'el': 'L',     # One-to-one
    'em': 'M',     # One-to-one
    'en': 'N',     # One-to-one
    'eng': 'NG',   # One-to-one
    'er': 'ER',    # One-to-one
    'ey': 'EY',    # One-to-one
    'f': 'F',      # One-to-one
    'g': 'G',      # One-to-one
    'gcl': 'G',    # Many-to-one (g, gcl -> G)
    'hh': 'HH',    # One-to-one
    'hv': 'HH',    # One-to-one (approximated to HH)
    'ih': 'IH',    # One-to-one
    'ix': 'IH',    # One-to-one (approximated to IH)
    'iy': 'IY',    # One-to-one
    'jh': 'JH',    # One-to-one
    'k': 'K',      # One-to-one
    'kcl': 'K',    # Many-to-one (k, kcl -> K)
    'l': 'L',      # One-to-one
    'm': 'M',      # One-to-one
    'n': 'N',      # One-to-one
    'ng': 'NG',    # One-to-one
    'nx': 'N',     # One-to-one (approximated to N)
    'ow': 'OW',    # One-to-one
    'oy': 'OY',    # One-to-one
    'p': 'P',      # One-to-one
    'pcl': 'P',    # Many-to-one (p, pcl -> P)
    'q': 'K',      # One-to-one (approximated to K)
    'r': 'R',      # One-to-one
    's': 'S',      # One-to-one
    'sh': 'SH',    # One-to-one
    't': 'T',      # One-to-one
    'tcl': 'T',    # Many-to-one (t, tcl -> T)
    'th': 'TH',    # One-to-one
    'uh': 'UH',    # One-to-one
    'uw': 'UW',    # One-to-one
    'ux': 'UW',    # One-to-one (approximated to UW)
    'v': 'V',      # One-to-one
    'w': 'W',      # One-to-one
    'y': 'Y',      # One-to-one
    'z': 'Z',      # One-to-one
    'zh': 'ZH',    # One-to-one
    'sil': 'SIL',  # Special symbol for silence
    'h#': 'SIL',   # Special symbol for silence
    'epi': 'SIL',  # Special symbol for silence
    'pau': 'SIL'   # Special symbol for pause
}

ARPABET = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
ARPABET_VOICELESS = ["P", "T", "CH", "F", "TH", "S", "SH", "HH"]
