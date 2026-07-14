"""
Step 1: Clean Data
==================
Normalizes point names in the training data while preserving word boundaries.

Text column processing (the same function is used at inference time, so
training and deployment always see identically normalized text):
  1. BOM/whitespace stripped.
  2. Niagara slot-path escapes decoded ($20 -> space, $2d -> '-', $2f -> '/').
  3. Any non-alphanumeric run becomes a separator.
  4. camelCase / acronym boundaries split into words.
  5. Lowercase.
  6. Equipment-index digits are stripped from tokens (site-specific numbering
     carries no point semantics): "ahu13" -> "ahu", "cwp01flt" -> "cwp flt",
     standalone digits are dropped. Chemical species, refrigerants (co2,
     pm25, r134a, ...) and single-letter+digit tokens (l1/l2/l3 phases) are
     preserved intact. Names that are ALL indices (e.g. "AV 12") fall back to
     their pre-stripping form rather than normalizing to nothing.
  7. Glued compounds with no case/separator/digit boundary ("DATEMP") are
     split by peeling known semantic tails right-to-left until a known head
     remains ("datemp" -> "da temp", "zntempminspt" -> "zn temp min spt").
     Both pieces must be in the curated lexicon; anything unknown passes
     through untouched, so English words ("compress", "thermostat") and
     novel tokens are never mangled.

  "Zone Temperature"   -> "zone temperature"
  "ZN-T"               -> "zn t"
  "CDK_DMPR_STATUS"    -> "cdk dmpr status"
  "zoneCO2Sp"          -> "zone co2 sp"
  "NGT$20CLG$20STPT"   -> "ngt clg stpt"
  "AHU13_SaTmp"        -> "ahu sa tmp"
  "L1_Current"         -> "l1 current"
  "DATEMP"             -> "da temp"

Target column: only stripped of BOM/whitespace (labels stay camelCase).

Input:  data/train_all.csv
Output: data/cleaned_data.csv
"""

import pandas as pd
import re
import sys


# Gas / particulate / refrigerant tokens from eo66 definitions and markers:
# their digits are meaning, not equipment numbering.
SPECIES = {
    'co2', 'so2', 'no2', 'n2', 'o2', 'o3', 'h2', 'h2s', 'ch4', 'cl2',
    'pm1', 'pm2', 'pm4', 'pm10', 'pm25',
    'r11', 'r22', 'r114', 'r123', 'r125', 'r134', 'r134a', 'r410', 'r410a',
}
# Unambiguous species suffixes so compounds survive too (rco2 = return CO2).
# Short suffixes like n2/h2 are excluded: fan2/ah2 are equipment numbering.
SPECIES_SUFFIXES = ('co2', 'so2', 'no2', 'pm25', 'pm10')


def _strip_index_digits(token: str) -> str:
    """Drop equipment-index digits from a token, keeping semantic ones."""
    if token in SPECIES or token.endswith(SPECIES_SUFFIXES):
        return token
    # Single letter + digits (l1/l2/l3 electrical phases, a1, b2): too
    # ambiguous to strip -- the digit may be the only discriminator.
    if re.fullmatch(r'[a-z]\d+', token):
        return token
    parts = re.split(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])', token)
    kept = []
    for part in parts:
        if not part.isdigit():
            kept.append(part)
        # Glued species survive: returnco2level -> returnco2 level
        elif kept and ((kept[-1] + part) in SPECIES
                       or (kept[-1] + part).endswith(SPECIES_SUFFIXES)):
            kept[-1] += part
    return ' '.join(kept)


# --- Glued-compound de-glue -------------------------------------------------
# All-caps glued names (DATEMP, RATEMPSP) survive every boundary rule above as
# a single token; the model's SentencePiece then carves them along English
# words ("datemp" -> date+mp) and the signal is destroyed. Peel known semantic
# tails right-to-left until a known head remains; if any piece is unknown the
# token is left untouched. Both sets are curated (seeded from
# data/abbreviations.csv) -- precision over recall by design.
DEGLUE_TAILS = tuple(sorted((
    'temp', 'tmp', 'sp', 'spt', 'stpt', 'sts', 'stat', 'cmd', 'fb', 'pos',
    'alm', 'min', 'max', 'press', 'prs', 'hum', 'flow', 'vlv', 'dmpr', 'spd',
    'co2',
), key=len, reverse=True))
DEGLUE_HEADS = frozenset({
    # air streams / locations
    'da', 'sa', 'ra', 'ma', 'oa', 'ea', 'zn', 'rm',
    # water loops
    'chw', 'hw', 'cw', 'cdw', 'hhw', 'twr', 'cond', 'evap', 'pri', 'sec',
    # equipment
    'ahu', 'vav', 'fcu', 'rtu', 'cuh', 'ef', 'sf', 'rf', 'ct', 'blr', 'chlr',
    'hx', 'cwp', 'chwp', 'hwp', 'pmp', 'fan', 'comp',
    # acronym compounds (DATSP -> dat sp)
    'dat', 'sat', 'rat', 'mat', 'oat', 'znt',
    # modifiers / directions
    'hi', 'lo', 'min', 'max', 'occ', 'clg', 'htg', 'frz', 'econ', 'sup', 'ret',
    # measurements (TEMPSP -> temp sp); also terminate multi-tail peels
    'temp', 'tmp', 'flow', 'press', 'hum', 'vlv', 'dmpr', 'spd',
})


def _deglue(token: str) -> list:
    """Split a glued compound into lexicon words; unknown tokens unchanged."""
    if token in DEGLUE_HEADS or len(token) < 4:
        return [token]
    parts, rest = [], token
    while rest not in DEGLUE_HEADS:
        for tail in DEGLUE_TAILS:
            if rest.endswith(tail) and len(rest) > len(tail):
                parts.insert(0, tail)
                rest = rest[:-len(tail)]
                break
        else:
            return [token]
    return [rest] + parts


def normalize_text(s: str) -> str:
    """Normalize a raw point name into lowercase space-separated words."""
    s = s.replace('\ufeff', '').strip()
    # Niagara slot-path hex escapes ($20 = space, $2d = '-', $2e = '.')
    s = re.sub(r'\$([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), s)
    # Every non-word run is a separator (covers _-./:#$() and any character
    # produced by escape decoding); unicode letters survive so lookalike
    # tokens in vendor exports are not destroyed
    s = re.sub(r'[\W_]+', ' ', s)
    # camelCase boundary: lowercase/digit followed by uppercase
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', s)
    # Acronym boundary: "CDKDamper" -> "CDK Damper" (keeps "CO2" intact)
    s = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    tokens = []
    for t in s.split():
        for piece in _strip_index_digits(t).split():
            tokens.extend(_deglue(piece))
    cleaned = ' '.join(tokens)
    # All-index names ("AV 12") must not normalize to nothing
    return cleaned if cleaned else s


def clean_data(input_file='data/train_all.csv', output_file='data/cleaned_data.csv'):
    """Normalize text column, strip target column."""
    try:
        print(f"Loading {input_file}...")
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        df['text'] = df['text'].astype(str).map(normalize_text)
        df['target'] = df['target'].astype(str).str.replace('\ufeff', '').str.strip()

        # Drop rows that normalized to nothing
        before = len(df)
        df = df[(df['text'].str.len() > 0) & (df['target'].str.len() > 0)]
        if before - len(df) > 0:
            print(f"Removed {before - len(df)} rows with empty text/target")

        df.to_csv(output_file, index=False)
        print(f"Saved cleaned data to {output_file}")

        return df

    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        sys.exit(1)


if __name__ == "__main__":
    clean_data()
