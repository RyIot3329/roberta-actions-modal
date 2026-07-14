# Tests for scripts/clean_data.py normalize_text, focused on the de-glue step.
#
# Run with pytest, or standalone: python3 tests/test_normalize_text.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from clean_data import normalize_text, _deglue  # noqa: E402

# Pre-existing documented behaviors (mirrors the ai-inference golden contract):
# the de-glue must not disturb any of them.
LEGACY = [
    ("Zone Temperature", "zone temperature"),
    ("ZN-T", "zn t"),
    ("CDK_DMPR_STATUS", "cdk dmpr status"),
    ("zoneCO2Sp", "zone co2 sp"),
    ("NGT$20CLG$20STPT", "ngt clg stpt"),
    ("AHU13_SaTmp", "ahu sa tmp"),
    ("L1_Current", "l1 current"),
    ("AV 12", "av"),
    ("AV 00", "av"),
    ("CWP01Flt", "cwp flt"),
    ("returnco2level", "returnco2 level"),
]

# De-glue: all-caps glued compounds with no case/separator/digit boundary.
DEGLUE = [
    ("DATEMP", "da temp"),
    ("DATMP", "da tmp"),
    ("SATEMP", "sa temp"),
    ("RATEMP", "ra temp"),
    ("MATEMP", "ma temp"),
    ("OATEMP", "oa temp"),
    ("ZNTEMP", "zn temp"),
    ("RATEMPSP", "ra temp sp"),          # multi-tail peel
    ("ZNTEMPMINSPT", "zn temp min spt"),  # three tails
    ("TEMPSP", "temp sp"),                # measurement word as head
    ("FLOWSP", "flow sp"),
    ("FRZSTAT", "frz stat"),
    ("CHWVLV", "chw vlv"),
    ("HWVLVCMD", "hw vlv cmd"),
    ("OADMPRPOS", "oa dmpr pos"),
    ("FANSTS", "fan sts"),
    ("EFSPD", "ef spd"),
    ("MINFLOW", "min flow"),
    ("CHWSP", "chw sp"),
    ("DATEMP01", "da temp"),              # index stripping composes with de-glue
    ("VAV_A1_01DaTemp", "vav a1 da temp"),
]

# Tokens the de-glue must NOT touch: English words whose suffix collides with
# a tail but whose remainder is not a known head, unknown heads, and
# multi-head glue (left-side peeling is deliberately not attempted, so the
# memorized train mapping for e.g. 'maxdatemp setpt' is preserved).
UNTOUCHED = [
    "compress", "thermostat", "overflow", "attempt", "temperature",
    "pressure", "humidity", "status", "setpoint", "position", "alarm",
    "command", "airflow", "wisp", "grasp", "tstat", "vitamin", "climax",
    "admin", "stpt", "sp", "temp", "maxdatemp", "hxtemp01x", "xk", "qqz",
]
# Known accepted collision: bare 'rasp' reads as return-air sp -- in BAS
# exports that is the correct domain reading, and the word never occurs
# alone as a point name.
ACCEPTED_COLLISIONS = [("rasp", "ra sp")]


def test_legacy_behaviors_unchanged():
    for raw, expected in LEGACY:
        assert normalize_text(raw) == expected, raw


def test_deglue():
    for raw, expected in DEGLUE:
        assert normalize_text(raw) == expected, raw


def test_unknown_tokens_untouched():
    for token in UNTOUCHED:
        assert _deglue(token) == [token], token


def test_accepted_collisions():
    for raw, expected in ACCEPTED_COLLISIONS:
        assert normalize_text(raw) == expected, raw


def test_idempotent():
    for raw, _ in LEGACY + DEGLUE:
        once = normalize_text(raw)
        assert normalize_text(once) == once, raw


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
