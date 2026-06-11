"""
Train-time augmentation for the resolved training pool.
========================================================

Emits perturbed variants of training texts so the model learns invariance
to the noise dimensions that separate sites: equipment-stem prefixes,
dropped tokens, and token order. Runs INSIDE convert_to_jsonl.py after
conflict resolution, so every variant inherits its source text's final
adjudicated label and is train-only by construction.

All ops work in post-normalization token space and only ever produce
strings that normalize_text maps to themselves (lowercase alpha tokens,
no digits -- digit-bearing prepends would be rewritten by the index
stripper at inference).

Decontamination: a variant is dropped if it equals (a) any organic train
text (it could carry a different resolved label and would bypass conflict
resolution), (b) any held-out val/test text (prepend-augmentation can
manufacture exactly the strings held-out names are made of, which would
flip their seen_in_train flags and corrupt the sliced reporting),
(c) any label_overrides.csv text (including DROP rows -- augmentation must
not resurrect an explicitly dropped string), or (d) a duplicate of an
already-emitted variant.

Determinism: per-text RNG seeded by sha256(seed|text); adding or removing
one pool text never reshuffles another text's variants, keeping committed
train.jsonl diffs minimal.
"""

import hashlib
import random

from clean_data import normalize_text

# Generic equipment stems that do not name an eo66-classed equipment family.
# Class-bearing stems (blr, boiler, ch, chiller, twr, hp, pmp, ac, unit, ...)
# are deliberately excluded: prepending them can change a text's true class.
EQUIPMENT_STEMS = [
    'ahu', 'rtu', 'fcu', 'vav', 'mau', 'doas', 'crac',
    'erv', 'hru', 'uh', 'cuh', 'fpb', 'cu', 'hx',
]
# Real site codes observed in TRAINING sites only (cbz = Integ01).
SITE_CODES = ['cbz']
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def _rng_for(seed: int, text: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{text}".encode()).hexdigest()
    return random.Random(int(digest, 16) % (2 ** 63))


def synth_site_codes(seed: int, corpus_vocab: set, count: int = 8) -> list:
    """Deterministic 2-3 letter pseudo site codes that collide with nothing
    meaningful (filtered against every token seen anywhere in the corpus)."""
    rng = random.Random(seed)
    codes = []
    attempts = 0
    while len(codes) < count and attempts < 1000:
        attempts += 1
        code = ''.join(rng.choice(ALPHABET) for _ in range(rng.choice((2, 3))))
        if code not in corpus_vocab and code not in codes:
            codes.append(code)
    return codes


def _variants_wanted(multiplier: float, seed: int, text: str) -> int:
    base = int(multiplier)
    frac = multiplier - base
    digest = hashlib.sha256(f"{seed}|alloc|{text}".encode()).hexdigest()
    return base + (1 if (int(digest, 16) % 1000) < frac * 1000 else 0)


def augment_pool(resolved: dict, heldout_texts: set, override_texts: set, cfg: dict):
    """
    Generate augmented (text, label) pairs from the resolved training pool.

    resolved: {text: label} after conflict resolution and overrides.
    Returns (variants list of (text, label), stats dict).
    """
    multiplier = float(cfg.get('augment_multiplier', 0) or 0)
    stats = {
        'augmented_texts': 0,
        'aug_dropped_train_collision': 0,
        'aug_dropped_heldout_collision': 0,
        'aug_dropped_override': 0,
        'aug_dropped_dup': 0,
        'aug_dropped_denormalized': 0,
    }
    if multiplier <= 0:
        return [], stats

    seed = int(cfg.get('augment_seed', 1337))
    dropout_p = float(cfg.get('dropout_p', 0.1))

    corpus_vocab = {tok for text in resolved for tok in text.split()}
    corpus_vocab |= {tok for text in heldout_texts for tok in text.split()}
    corpus_vocab |= set(EQUIPMENT_STEMS)
    prefixes = EQUIPMENT_STEMS + SITE_CODES + synth_site_codes(seed, corpus_vocab)

    organic = set(resolved)
    emitted = set()
    variants = []

    def perturb(rng, tokens):
        roll = rng.random()
        if roll < 0.5:  # prepend equipment stem / site code
            return [rng.choice(prefixes)] + tokens
        if roll < 0.8 and len(tokens) >= 2:  # token dropout
            kept = [t for t in tokens if rng.random() > dropout_p]
            return kept if kept else [rng.choice(tokens)]
        if len(tokens) >= 2:  # adjacent swap
            i = rng.randrange(len(tokens) - 1)
            swapped = tokens[:]
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            return swapped
        return [rng.choice(prefixes)] + tokens  # single-token fallback

    for text, label in sorted(resolved.items()):
        wanted = _variants_wanted(multiplier, seed, text)
        if wanted == 0:
            continue
        rng = _rng_for(seed, text)
        tokens = text.split()
        produced = 0
        for _ in range(wanted * 6):  # bounded attempts per variant
            if produced >= wanted:
                break
            candidate = ' '.join(perturb(rng, tokens))
            if candidate == text:
                continue
            # Train must only contain strings inference can produce: variants
            # of digit-bearing fallback texts (e.g. "12") fail this
            if normalize_text(candidate) != candidate:
                stats['aug_dropped_denormalized'] += 1
                continue
            if candidate in organic:
                stats['aug_dropped_train_collision'] += 1
            elif candidate in heldout_texts:
                stats['aug_dropped_heldout_collision'] += 1
            elif candidate in override_texts:
                stats['aug_dropped_override'] += 1
            elif candidate in emitted:
                stats['aug_dropped_dup'] += 1
            else:
                emitted.add(candidate)
                variants.append((candidate, label))
                produced += 1

    stats['augmented_texts'] = len(variants)
    assert not (emitted & heldout_texts), "augmentation leaked a held-out text"
    assert not (emitted & organic), "augmentation duplicated an organic train text"
    return variants, stats
