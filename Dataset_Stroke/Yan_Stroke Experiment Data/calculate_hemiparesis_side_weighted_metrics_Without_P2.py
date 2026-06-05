from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
RESULTS_XLSX = "Results_of_MHS.xlsx"
HEMIPARESIS_XLSX = "Hemiparesis Side.xlsx"

SENSOR_CONFIG_NAME = "in_S1-S2_to_S4-S5"

# For the current LSTM_Ses/TCN_Ses/SHRED_Ses output target sensors.
# In this dataset S1/S4 are left-side sensors, and S2/S5 are right-side sensors.
TARGET_SENSOR_SIDE = {
    "S1": "left",
    "S2": "right",
    "S4": "left",
    "S5": "right",
}

OUTPUT_DIR = "outputs/hemiparesis_side_weighted_metrics"
OUTPUT_SUBDIR_SUFFIX = "_Without_P2"
EXCLUDED_SUBJECTS = {"P2"}

SUMMARY_METHOD_ORDER = {"LSTM": 0, "SHRED": 1, "TCN": 2}
SUMMARY_MODALITY_ORDER = {"acc": 0, "gyro": 1}
SUMMARY_SIDE_GROUP_ORDER = {"affected": 0, "unaffected": 1}


# Prefer the new all-method batch output. If it does not exist yet, fall back to:
#   - LSTM rows parsed from Results_of_MHS.xlsx
#   - TCN/SHRED rows from the previously generated TCN/SHRED detail CSV
NEW_ALL_METHOD_DETAIL_CSV = (
    f"outputs/ses_models_from_mhs/Results_of_MHS_{SENSOR_CONFIG_NAME}_LSTM_TCN_SHRED/"
    f"ses_models_{SENSOR_CONFIG_NAME}_weighted_metrics_detail.csv"
)
LEGACY_TCN_SHRED_DETAIL_CSV = (
    "outputs/tcn_shred_from_mhs/Results_of_MHS_TCN_SHRED/"
    "tcn_shred_weighted_metrics_detail.csv"
)


XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NOTEBOOK_DIR = Path(__file__).resolve().parent


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = NOTEBOOK_DIR / path
    return path


def _col_to_idx(cell_ref: str) -> int:
    col = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for ch in col:
        idx = idx * 26 + ord(ch.upper()) - 64
    return idx - 1


def _shared_string_text(si: ET.Element) -> str:
    return "".join(node.text or "" for node in si.findall(".//m:t", XLSX_NS))


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Any:
    value_node = cell.find("m:v", XLSX_NS)
    if value_node is None:
        return ""

    text = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(text)]

    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text


def _read_xlsx_rows(xlsx_path: Path) -> list[list[Any]]:
    with ZipFile(xlsx_path) as archive:
        shared_strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared_strings = [
            _shared_string_text(si)
            for si in shared_strings_root.findall("m:si", XLSX_NS)
        ]

        worksheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in worksheet_root.findall(".//m:sheetData/m:row", XLSX_NS):
            values_by_col = {}
            for cell in row.findall("m:c", XLSX_NS):
                values_by_col[_col_to_idx(cell.attrib["r"])] = _cell_value(cell, shared_strings)
            if values_by_col:
                max_col = max(values_by_col)
                rows.append([values_by_col.get(idx, "") for idx in range(max_col + 1)])
    return rows


def _parse_session_list(text: str) -> list[int]:
    session_ids = []
    for token in re.split(r"[,，、\s]+", text.strip()):
        if not token:
            continue

        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", token)
        if range_match:
            start, end = map(int, range_match.groups())
            step = 1 if end >= start else -1
            session_ids.extend(range(start, end + step, step))
            continue

        if re.fullmatch(r"\d+", token):
            session_ids.append(int(token))

    unique_ids = []
    seen = set()
    for session_id in session_ids:
        if session_id not in seen:
            unique_ids.append(session_id)
            seen.add(session_id)
    return unique_ids


def _parse_training_note(note: str) -> tuple[list[int], list[int] | None]:
    match = re.search(
        r"训练\s*Session\s*[:：]\s*(?P<train>.*?)(?:Test\s*Session\s*[:：]\s*(?P<test>.*))?$",
        note,
        flags=re.IGNORECASE,
    )
    if not match:
        return [], None

    train_ids = _parse_session_list(match.group("train") or "")
    note_test_text = match.group("test")
    note_test_ids = _parse_session_list(note_test_text) if note_test_text else None
    return train_ids, note_test_ids


def _normalize_subject_id(value: Any) -> str | None:
    text = str(value).strip()
    match = re.fullmatch(r"[PS](\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"P{int(match.group(1))}"
    return None


def _normalize_side(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text.startswith("right") or text in {"r", "右", "right(r)"}:
        return "right"
    if text.startswith("left") or text in {"l", "左", "left(l)"}:
        return "left"
    return None


def _parse_hemiparesis_sides(xlsx_path: Path) -> pd.DataFrame:
    rows = _read_xlsx_rows(xlsx_path)
    parsed = []
    for row in rows[1:]:
        if len(row) < 2:
            continue

        subject = _normalize_subject_id(row[0])
        side = _normalize_side(row[1])
        if subject is None or side is None:
            continue

        parsed.append(
            {
                "Subject": subject,
                "Hemiparesis_Side": side,
            }
        )

    result = pd.DataFrame(parsed).drop_duplicates(subset=["Subject"])
    if result.empty:
        raise ValueError(f"No hemiparesis side rows were parsed from {xlsx_path}")
    return result


def _parse_lstm_detail_from_results_xlsx(xlsx_path: Path) -> pd.DataFrame:
    rows = _read_xlsx_rows(xlsx_path)
    detail_rows = []
    session_notes_by_subject = {}

    for row in rows:
        subject = _normalize_subject_id(row[1] if len(row) > 1 else "")
        if subject is None:
            continue

        for value in row:
            if isinstance(value, str) and "训练" in value and "Session" in value:
                session_notes_by_subject[subject] = value

    for row in rows:
        subject = _normalize_subject_id(row[1] if len(row) > 1 else "")
        if subject is None:
            continue

        if len(row) < 10 or str(row[3]).strip() not in TARGET_SENSOR_SIDE:
            continue

        modality = str(row[4]).strip()
        if modality not in {"acc", "gyro"}:
            continue

        test_session_ids = _parse_session_list(str(row[2]))
        train_session_ids, _ = _parse_training_note(session_notes_by_subject.get(subject, ""))
        train_session_ids = [sid for sid in train_session_ids if sid not in set(test_session_ids)]

        detail_rows.append(
            {
                "Method": "LSTM",
                "Subject": subject,
                "Sensor_Config": SENSOR_CONFIG_NAME,
                "Train_Session_IDS": ",".join(map(str, train_session_ids)),
                "Test_Session_IDS": ",".join(map(str, test_session_ids)),
                "SkippedExisting": "manual_xlsx",
                "sensor": str(row[3]).strip(),
                "modality": modality,
                "channels": str(row[5]) if len(row) > 5 else "",
                "RMSE_weight": float(row[6]),
                "PCC_weight": float(row[7]),
                "weighted_RMSE": float(row[8]),
                "weighted_PCC": float(row[9]),
            }
        )

    result = pd.DataFrame(detail_rows)
    if result.empty:
        raise ValueError(f"No LSTM detail metric rows were parsed from {xlsx_path}")
    return result


def _load_metric_detail() -> pd.DataFrame:
    new_detail_path = _resolve_path(NEW_ALL_METHOD_DETAIL_CSV)
    if new_detail_path.exists():
        print(f"Using all-method detail CSV: {new_detail_path}")
        detail = pd.read_csv(new_detail_path)
    else:
        frames = [_parse_lstm_detail_from_results_xlsx(_resolve_path(RESULTS_XLSX))]
        legacy_path = _resolve_path(LEGACY_TCN_SHRED_DETAIL_CSV)
        if legacy_path.exists():
            print(f"Using legacy TCN/SHRED detail CSV: {legacy_path}")
            frames.append(pd.read_csv(legacy_path))
        else:
            print(f"Legacy TCN/SHRED detail CSV not found: {legacy_path}")

        detail = pd.concat(frames, ignore_index=True)

    required = {"Method", "Subject", "sensor", "modality", "RMSE_weight", "PCC_weight", "weighted_RMSE", "weighted_PCC"}
    missing = required.difference(detail.columns)
    if missing:
        raise ValueError(f"Metric detail data is missing required columns: {sorted(missing)}")

    detail["Subject"] = detail["Subject"].map(_normalize_subject_id)
    detail = detail[detail["Subject"].notna()].copy()
    if "Sensor_Config" not in detail.columns:
        detail["Sensor_Config"] = SENSOR_CONFIG_NAME
    else:
        detail["Sensor_Config"] = detail["Sensor_Config"].fillna(SENSOR_CONFIG_NAME)
    detail["sensor"] = detail["sensor"].astype(str).str.strip()
    detail["modality"] = detail["modality"].astype(str).str.strip()
    detail = detail[detail["sensor"].isin(TARGET_SENSOR_SIDE)].copy()
    return detail


def _weighted_rmse(values: pd.DataFrame) -> tuple[float, int]:
    rmse = pd.to_numeric(values["weighted_RMSE"], errors="coerce")
    weight = pd.to_numeric(values["RMSE_weight"], errors="coerce")
    mask = rmse.notna() & weight.notna() & (weight > 0)
    if not mask.any():
        return np.nan, 0
    weight_sum = float(weight[mask].sum())
    return float(np.sqrt(((rmse[mask] ** 2) * weight[mask]).sum() / weight_sum)), int(weight_sum)


def _weighted_pcc(values: pd.DataFrame) -> tuple[float, int]:
    pcc = pd.to_numeric(values["weighted_PCC"], errors="coerce")
    weight = pd.to_numeric(values["PCC_weight"], errors="coerce")
    mask = pcc.notna() & weight.notna() & (weight > 0)
    if not mask.any():
        return np.nan, 0
    weight_sum = float(weight[mask].sum())
    return float((pcc[mask] * weight[mask]).sum() / weight_sum), int(weight_sum)


def _side_detail(metric_detail: pd.DataFrame, hemi_sides: pd.DataFrame) -> pd.DataFrame:
    merged = metric_detail.merge(hemi_sides, on="Subject", how="left")
    missing_side = sorted(merged.loc[merged["Hemiparesis_Side"].isna(), "Subject"].dropna().unique())
    if missing_side:
        print(f"Warning: missing hemiparesis side for subjects: {missing_side}")

    merged["Sensor_Side"] = merged["sensor"].map(TARGET_SENSOR_SIDE)
    merged["Side_Group"] = np.where(
        merged["Sensor_Side"] == merged["Hemiparesis_Side"],
        "affected",
        "unaffected",
    )
    return merged


def _aggregate_by(columns: list[str], detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        rmse, rmse_weight = _weighted_rmse(group)
        pcc, pcc_weight = _weighted_pcc(group)
        row = {column: value for column, value in zip(columns, keys)}
        row.update(
            {
                "RMSE_weight": rmse_weight,
                "PCC_weight": pcc_weight,
                "weighted_RMSE": rmse,
                "weighted_PCC": pcc,
                "n_rows": len(group),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _format_mean_pm(mean_value: float, sd_value: float) -> str:
    if pd.isna(mean_value) or pd.isna(sd_value):
        return ""
    return f"{mean_value:.4f}±{sd_value:.4f}"


def _subject_mean_sd_by(columns: list[str], subject_side: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in subject_side.groupby(columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        rmse = pd.to_numeric(group["weighted_RMSE"], errors="coerce").dropna()
        pcc = pd.to_numeric(group["weighted_PCC"], errors="coerce").dropna()
        rmse_mean = float(rmse.mean()) if not rmse.empty else np.nan
        pcc_mean = float(pcc.mean()) if not pcc.empty else np.nan
        rmse_sd = float(rmse.std(ddof=1)) if len(rmse) > 1 else 0.0 if len(rmse) == 1 else np.nan
        pcc_sd = float(pcc.std(ddof=1)) if len(pcc) > 1 else 0.0 if len(pcc) == 1 else np.nan

        row = {column: value for column, value in zip(columns, keys)}
        row.update(
            {
                "subject_n": int(group["Subject"].nunique()) if "Subject" in group.columns else len(group),
                "weighted_RMSE_subject_mean": rmse_mean,
                "weighted_RMSE_subject_sd": rmse_sd,
                "weighted_RMSE_mean_pm_sd": _format_mean_pm(rmse_mean, rmse_sd),
                "weighted_PCC_subject_mean": pcc_mean,
                "weighted_PCC_subject_sd": pcc_sd,
                "weighted_PCC_mean_pm_sd": _format_mean_pm(pcc_mean, pcc_sd),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _side_acc_gyro_pcc(all_subject_weighted: pd.DataFrame, per_subject_long: pd.DataFrame) -> pd.DataFrame:
    side_columns = [
        column
        for column in ["Method", "Sensor_Config", "Side_Group"]
        if column in all_subject_weighted.columns
    ]
    rows = []
    for keys, group in all_subject_weighted.groupby(side_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        pcc = pd.to_numeric(group["weighted_PCC"], errors="coerce")
        weight = pd.to_numeric(group["PCC_weight"], errors="coerce")
        mask = pcc.notna() & weight.notna() & (weight > 0)
        side_pcc = np.nan
        if mask.any():
            side_pcc = float((pcc[mask] * weight[mask]).sum() / weight[mask].sum())

        row = {column: value for column, value in zip(side_columns, keys)}
        row["side_acc_gyro_weighted_PCC"] = side_pcc
        rows.append(row)

    all_subject_side_pcc = pd.DataFrame(rows)

    subject_columns = [
        column
        for column in ["Method", "Subject", "Sensor_Config", "Side_Group"]
        if column in per_subject_long.columns
    ]
    subject_side_rows = []
    for keys, group in per_subject_long.groupby(subject_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        pcc = pd.to_numeric(group["weighted_PCC"], errors="coerce")
        weight = pd.to_numeric(group["PCC_weight"], errors="coerce")
        mask = pcc.notna() & weight.notna() & (weight > 0)
        side_pcc = np.nan
        if mask.any():
            side_pcc = float((pcc[mask] * weight[mask]).sum() / weight[mask].sum())

        row = {column: value for column, value in zip(subject_columns, keys)}
        row["side_acc_gyro_weighted_PCC"] = side_pcc
        subject_side_rows.append(row)

    subject_side_pcc = pd.DataFrame(subject_side_rows)
    if subject_side_pcc.empty:
        return all_subject_side_pcc

    rows = []
    for keys, group in subject_side_pcc.groupby(side_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        pcc = pd.to_numeric(group["side_acc_gyro_weighted_PCC"], errors="coerce").dropna()
        pcc_mean = float(pcc.mean()) if not pcc.empty else np.nan
        pcc_sd = float(pcc.std(ddof=1)) if len(pcc) > 1 else 0.0 if len(pcc) == 1 else np.nan

        row = {column: value for column, value in zip(side_columns, keys)}
        row.update(
            {
                "side_acc_gyro_subject_n": int(group["Subject"].nunique()) if "Subject" in group.columns else len(pcc),
                "side_acc_gyro_weighted_PCC_subject_mean": pcc_mean,
                "side_acc_gyro_weighted_PCC_subject_sd": pcc_sd,
                "side_acc_gyro_weighted_PCC_mean_pm_sd": _format_mean_pm(pcc_mean, pcc_sd),
            }
        )
        rows.append(row)

    return all_subject_side_pcc.merge(pd.DataFrame(rows), on=side_columns, how="left")


def _sort_all_subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    sensor_config_order = {config: idx for idx, config in enumerate(pd.unique(result["Sensor_Config"]))}

    result["_method_order"] = result["Method"].map(SUMMARY_METHOD_ORDER).fillna(len(SUMMARY_METHOD_ORDER))
    result["_sensor_config_order"] = result["Sensor_Config"].map(sensor_config_order).fillna(len(sensor_config_order))
    result["_modality_order"] = result["modality"].map(SUMMARY_MODALITY_ORDER).fillna(len(SUMMARY_MODALITY_ORDER))
    result["_side_group_order"] = result["Side_Group"].map(SUMMARY_SIDE_GROUP_ORDER).fillna(len(SUMMARY_SIDE_GROUP_ORDER))
    result = result.sort_values(
        ["_method_order", "_sensor_config_order", "_modality_order", "_side_group_order"],
        kind="mergesort",
    )
    return result.drop(
        columns=["_method_order", "_sensor_config_order", "_modality_order", "_side_group_order"]
    ).reset_index(drop=True)


def _format_float_columns(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    result = df.copy()
    float_columns = result.select_dtypes(include=["float"]).columns
    for column in float_columns:
        result[column] = result[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    return result


def _summary_wide(subject_side: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, subject), group in subject_side.groupby(["Method", "Subject"], dropna=False):
        row = {"Method": method, "Subject": subject}
        if "Sensor_Config" in group.columns:
            row["Sensor_Config"] = group["Sensor_Config"].dropna().iloc[0] if group["Sensor_Config"].notna().any() else SENSOR_CONFIG_NAME
        if "Hemiparesis_Side" in group.columns:
            row["Hemiparesis_Side"] = group["Hemiparesis_Side"].dropna().iloc[0] if group["Hemiparesis_Side"].notna().any() else np.nan

        for _, item in group.iterrows():
            prefix = f"{item['Side_Group']}_{item['modality']}"
            row[f"{prefix}_RMSE"] = item["weighted_RMSE"]
            row[f"{prefix}_PCC"] = item["weighted_PCC"]
            row[f"{prefix}_RMSE_weight"] = item["RMSE_weight"]
            row[f"{prefix}_PCC_weight"] = item["PCC_weight"]
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    output_dir = _resolve_path(OUTPUT_DIR) / f"{SENSOR_CONFIG_NAME}{OUTPUT_SUBDIR_SUFFIX}"
    output_dir.mkdir(parents=True, exist_ok=True)

    hemi_sides = _parse_hemiparesis_sides(_resolve_path(HEMIPARESIS_XLSX))
    metric_detail = _load_metric_detail()
    hemi_sides = hemi_sides[~hemi_sides["Subject"].isin(EXCLUDED_SUBJECTS)].copy()
    metric_detail = metric_detail[~metric_detail["Subject"].isin(EXCLUDED_SUBJECTS)].copy()
    print(f"Excluded subjects: {sorted(EXCLUDED_SUBJECTS)}")
    side_detail = _side_detail(metric_detail, hemi_sides)

    side_detail_output = output_dir / f"{SENSOR_CONFIG_NAME}_metric_detail_with_side.csv"
    per_subject_long_output = output_dir / f"{SENSOR_CONFIG_NAME}_per_subject_side_weighted_long.csv"
    per_subject_wide_output = output_dir / f"{SENSOR_CONFIG_NAME}_per_subject_side_weighted_summary.csv"
    all_subject_output = output_dir / f"{SENSOR_CONFIG_NAME}_all_subject_side_weighted_summary.csv"

    side_detail.to_csv(side_detail_output, index=False)

    subject_columns = ["Method", "Subject", "Sensor_Config", "Hemiparesis_Side", "Side_Group", "modality"]
    subject_columns = [column for column in subject_columns if column in side_detail.columns]
    per_subject_long = _aggregate_by(subject_columns, side_detail)
    per_subject_long.to_csv(per_subject_long_output, index=False)

    per_subject_wide = _summary_wide(per_subject_long)
    per_subject_wide.to_csv(per_subject_wide_output, index=False)

    all_columns = ["Method", "Sensor_Config", "Side_Group", "modality"]
    all_columns = [column for column in all_columns if column in side_detail.columns]
    all_subject_weighted = _aggregate_by(all_columns, side_detail)
    side_pcc = _side_acc_gyro_pcc(all_subject_weighted, per_subject_long)
    subject_mean_sd = _subject_mean_sd_by(all_columns, per_subject_long)
    all_subject = all_subject_weighted.merge(subject_mean_sd, on=all_columns, how="left")
    all_subject = all_subject.merge(side_pcc, on=[column for column in ["Method", "Sensor_Config", "Side_Group"] if column in all_subject.columns], how="left")
    all_subject = _sort_all_subject_summary(all_subject)
    all_subject = _format_float_columns(all_subject, decimals=4)
    all_subject.to_csv(all_subject_output, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(f"Saved side detail CSV: {side_detail_output}")
    print(f"Saved per-subject side summary CSV: {per_subject_wide_output}")
    print(f"Saved all-subject side summary CSV: {all_subject_output}")
    print("\nAll-subject affected/unaffected weighted metrics:")
    print(all_subject.to_string(index=False))


if __name__ == "__main__":
    main()
