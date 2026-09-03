"""Tests for the context parsers in scripts/extract_real_data.py.

Run: python3 tests/test_extract_real_data_parsers.py   (or pytest tests/)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from extract_real_data import (CONTEXT_COLUMNS, FORBIDDEN_COLUMNS, FORMATS,  # noqa: E402
                               equip_from_slot_path, extract_frame,
                               object_type_from_bacnet_id, units_from_to_string,
                               value_kind_from_out)


def test_equip_from_slot_path_shapes():
    assert equip_from_slot_path(
        "slot:/Drivers/NiagaraNetwork/NH0037ZZ/Norris_Cotton/points/AHU/HVAC_01A/BldgPrs"
    ) == "AHU HVAC_01A"
    assert equip_from_slot_path(
        "slot:/Drivers/NiagaraNetwork/FLOOR_1/points/AC_1/INPUTS/SupplyTemperat") == "AC_1 INPUTS"
    assert equip_from_slot_path(
        "slot:/Drivers/BacnetNetwork/Mstp103/VAV1N_1_J013B002/points/Ins/InletA"
    ) == "VAV1N_1_J013B002 Ins"
    assert equip_from_slot_path("slot:/Drivers/NiagaraNetwork/R01/NH0037ZZ_Norris$20Cotton/Chl"
                                ) == "R01 NH0037ZZ_Norris$20Cotton"
    assert equip_from_slot_path("SupplyTemperature") == ""       # leaf only
    assert equip_from_slot_path("slot:/Drivers/NiagaraNetwork/points/Name") == ""
    assert equip_from_slot_path(None) == ""


def test_units_from_to_string():
    assert units_from_to_string("106.68 °F {ok} @ def") == "°F"
    assert units_from_to_string("-0.0052 in/wc {ok}") == "in/wc"
    assert units_from_to_string("794 cfm {ok}") == "cfm"
    assert units_from_to_string("0.00 % {ok} @ def") == "%"
    assert units_from_to_string("28265.10 in/wc {ok} @ 10") == "in/wc"
    assert units_from_to_string("OFF {ok} @ def") == "OFF"       # enum/boolean state text
    assert units_from_to_string("Enable {ok} @ 14") == "Enable"
    assert units_from_to_string("1.00 {ok} @ def") == ""        # number without a unit
    assert units_from_to_string(None) == ""


def test_value_kind_from_out():
    assert value_kind_from_out("true {ok} @ def") == "binary"
    assert value_kind_from_out("false {ok}") == "binary"
    assert value_kind_from_out("68.00 {ok}") == "analog"
    assert value_kind_from_out("-0.01 {ok}") == "analog"
    assert value_kind_from_out("Occupied {ok}") == "enum"
    assert value_kind_from_out("") == ""


def test_object_type():
    assert object_type_from_bacnet_id("AV0") == "AV"
    assert object_type_from_bacnet_id("BI3") == "BI"
    assert object_type_from_bacnet_id("msv12") == "MSV"
    assert object_type_from_bacnet_id(None) == ""


def _n4_frame():
    return pd.DataFrame({
        "proxyExt.pointId/BASpointName": [
            "slot:/Drivers/NiagaraNetwork/MCSJace06/points/AHU_1W/ChwVlv",
            "slot:/Drivers/NiagaraNetwork/MCSJace06/points/AHU_S/PressDmprFdbk",
            "slot:/Drivers/NiagaraNetwork/X/points/Chiller/ChlSts"],
        "pointTag/EO66": ["coolingValve", "intakeDamperFb", "chillerStatus"],
        "out": ["0.00 {ok}", "12.5 {ok} @ 2", "false {ok} @ def"],
        "To String": ["0.0 % {ok}", "12.5 % {ok} @ 2", "OFF {ok} @ def"],
        # post-mapping columns present in the export: must be ignored
        "parent.name/EquipName": ["AHU_1W", "AHU_S", "CLR_02"],
        "Slot Path": ["slot:/Drivers/NiagaraNetwork/R1/SUPERVISOR/a"] * 3,
    })


def test_n4_layout_context():
    out = extract_frame(_n4_frame())
    assert list(out.columns) == ["name"] + CONTEXT_COLUMNS + ["label_raw"]
    assert out["name"].tolist() == ["ChwVlv", "PressDmprFdbk", "ChlSts"]
    assert out["equip_path"].tolist() == ["MCSJace06 AHU_1W", "MCSJace06 AHU_S", "X Chiller"]
    assert out["units"].tolist() == ["%", "%", "OFF"]
    assert out["value_kind"].tolist() == ["analog", "analog", "binary"]
    assert (out["device_name"] == "").all() and (out["object_type"] == "").all()


def test_forbidden_columns_never_read():
    frame = _n4_frame()
    base = extract_frame(frame)
    frame2 = frame.copy()
    for c in FORBIDDEN_COLUMNS & set(frame2.columns):
        frame2[c] = "LEAKED_" + frame2[c].astype(str)
    assert extract_frame(frame2).equals(base)
    for fmt in FORMATS:
        used = {v for k, v in fmt.items() if k != "is_path"}
        assert not (used & FORBIDDEN_COLUMNS)


def test_bacnet_layout_context():
    frame = pd.DataFrame({
        "BACnet Device Name": ["VAV 2.2 [2518]", "Annex 1st Floor [1512]"],
        "BACnet Object ID": ["AV0", "BI7"],
        "BACnet Description": ["Heating Signal", None],
        "Bacnet Name": ["AV 00", "Fan Status"],
        "eo66Def": ["heatingDemand", "dischargeFanStatus"],
    })
    out = extract_frame(frame)
    assert out["name"].tolist() == ["AV 00", "Fan Status"]
    assert out["device_name"].tolist() == ["VAV 2.2 [2518]", "Annex 1st Floor [1512]"]
    assert out["object_type"].tolist() == ["AV", "BI"]
    assert out["value_kind"].tolist() == ["analog", "binary"]
    assert out["description"].tolist() == ["Heating Signal", ""]
    assert (out["equip_path"] == "").all() and (out["units"] == "").all()


def test_unknown_layout_returns_none():
    assert extract_frame(pd.DataFrame({"a": [1]})) is None


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
