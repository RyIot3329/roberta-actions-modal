"""Tests for the shared context builder (scripts/clean_data.py).

Run: python3 tests/test_clean_data_context.py   (or pytest tests/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from clean_data import (CONTEXT_FIELDS, CONTEXT_VERSION, build_context,  # noqa: E402
                        build_model_input, canon_unit, normalize_text)


def test_empty_context_is_name_only():
    assert build_context() == ""
    assert build_model_input("zone temp", "") == "zone temp"
    assert build_model_input("zone temp", None) == "zone temp"


def test_field_order_is_fixed():
    a = build_context(units="°F", equip="AHU_01/INPUTS", value_kind="analog")
    b = build_context(value_kind="analog", equip="AHU_01/INPUTS", units="°F")
    assert a == b == "eq ahu inputs | unit degf | kind analog"
    assert CONTEXT_FIELDS[0] == "eq"


def test_units_canon():
    assert canon_unit("°F") == "degf"
    assert canon_unit("deg F") == "degf"
    assert canon_unit("%") == "percent"
    assert canon_unit("in/wc") == "inwc"
    assert canon_unit("cfm") == "cfm"
    assert canon_unit("L/s") == "lps"
    assert canon_unit("") == ""
    assert canon_unit("furlongs/fortnight") == "furlongsfortnight"  # unknown: alnum only


def test_keep_subsets_fields():
    full = build_context(equip="AHU_01", description="Heating Signal", units="%",
                         value_kind="analog", object_type="AV", device="VAV 2.2 [2518]")
    assert full == ("eq ahu | desc heating signal | unit percent | kind analog | obj av | dev vav")
    assert build_context(equip="AHU_01", units="%", keep=["unit"]) == "unit percent"
    assert build_context(equip="AHU_01", units="%", keep=[]) == ""


def test_closed_vocabularies_and_species():
    assert build_context(value_kind="Boolean") == "kind binary"
    assert build_context(value_kind="true") == "kind binary"
    assert build_context(value_kind="number") == "kind analog"
    assert build_context(value_kind="String") == "kind enum"
    assert build_context(object_type="BI3") == "obj bi"
    assert build_context(object_type="multi-state-value") == "obj msv"
    assert build_context(description="Zone CO2 Level") == "desc zone co2 level"


def test_indices_and_escapes_stripped():
    assert build_context(equip="VAV1N_1_J013B002/Ins") == "eq vav n ins"
    assert build_context(equip="NH0037ZZ_Norris$20Cotton$20FB") == "eq nh zz norris cotton fb"


def test_model_input_is_idempotent_under_normalization():
    ctx = build_context(equip="AHU_01/INPUTS", units="°F", value_kind="analog")
    text = build_model_input("sa tmp", ctx)
    assert text == "sa tmp | eq ahu inputs | unit degf | kind analog"
    # every context token must survive normalize_text unchanged (no re-normalization
    # drift); only the field separators disappear
    assert normalize_text(ctx).split() == [t for t in ctx.split() if t != "|"]


def test_version_string():
    assert CONTEXT_VERSION == "1"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
