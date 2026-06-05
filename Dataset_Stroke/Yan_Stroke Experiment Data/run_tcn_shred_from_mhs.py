from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
RESULTS_XLSX = "Results_of_MHS.xlsx"
INPUT_SENSORS = ["S1", "S2"]
TARGET_SENSORS = ["S4", "S5"]


# ---------------------------------------------------------------------------
# Shared notebook settings
# ---------------------------------------------------------------------------
WINDOW_SIZE = 225
BATCH_SIZE = 32
VALIDATION_SIZE = 0.2
TRAIN_STRIDE = 32
VAL_STRIDE = 64
TEST_STRIDE = 64
EPOCHS = 100
LR = 0.001
TEST_WINDOW_IDX = 3


# ---------------------------------------------------------------------------
# Method-specific architecture settings
# ---------------------------------------------------------------------------
METHODS = ("LSTM", "TCN", "SHRED")

TCN_CHANNELS = (64, 128, 256)
TCN_KERNEL_SIZE = 5
TCN_DROPOUT = 0.2

SHRED_HIDDEN_SIZE = 64
SHRED_HIDDEN_LAYERS = 2
SHRED_DECODER_L1 = 350
SHRED_DECODER_L2 = 400
SHRED_DROPOUT = 0.1


# ---------------------------------------------------------------------------
# Batch-run settings
# ---------------------------------------------------------------------------
OUTPUT_BASE_DIR = "outputs/ses_models_from_mhs"
RUN_NAME = None
SUBJECT_FILTER = None  # Example: ["P1", "P4"]. Keep None to run all subjects in RESULTS_XLSX.
SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
CELL_TIMEOUT = -1
KERNEL_NAME = None


NOTEBOOK_FILES = {
    "LSTM": "LSTM_Ses.ipynb",
    "TCN": "TCN_Ses.ipynb",
    "SHRED": "SHRED_Ses.ipynb",
}

XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _find_project_python() -> Path | None:
    script_dir = Path(__file__).resolve().parent
    candidates = []
    for parent in (script_dir, *script_dir.parents):
        candidates.extend(
            [
                parent / "SHREDwyc" / "bin" / "python",
                parent / ".venv" / "bin" / "python",
                parent / "venv" / "bin" / "python",
            ]
        )

    current = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != current:
            return candidate
    return None


def _load_notebook_runner():
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor

        return nbformat, ExecutePreprocessor
    except ModuleNotFoundError:
        project_python = _find_project_python()
        if project_python is not None:
            os.execv(
                str(project_python),
                [str(project_python), str(Path(__file__).resolve()), *sys.argv[1:]],
            )
        raise SystemExit(
            "Missing nbformat/nbconvert. Run this script with the SHREDwyc Python "
            "environment, for example: /mnt/ssd1/wyc/SHREDwyc/bin/python run_tcn_shred_from_mhs.py"
        )


nbformat, ExecutePreprocessor = _load_notebook_runner()

import pandas as pd


NOTEBOOK_DIR = Path(__file__).resolve().parent


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = NOTEBOOK_DIR / path
    return path


def _sensor_config_name() -> str:
    input_text = "-".join(INPUT_SENSORS)
    target_text = "-".join(TARGET_SENSORS)
    return f"in_{input_text}_to_{target_text}"


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
        raise ValueError(f"Cannot parse training note: {note!r}")

    train_ids = _parse_session_list(match.group("train") or "")
    note_test_text = match.group("test")
    note_test_ids = _parse_session_list(note_test_text) if note_test_text else None
    return train_ids, note_test_ids


def _parse_mhs_subject_plan(xlsx_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _read_xlsx_rows(xlsx_path)
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        subject = row[1] if len(row) > 1 else ""
        if not isinstance(subject, str) or not re.fullmatch(r"P\d+", subject):
            continue

        entry = grouped.setdefault(subject, {"subject": subject, "test_ids": set(), "notes": []})
        if len(row) > 2 and row[2] != "":
            entry["test_ids"].add(int(row[2]))

        for value in row:
            if isinstance(value, str) and "训练" in value and "Session" in value:
                entry["notes"].append(value)

    plans = []
    warnings = []
    for subject in sorted(grouped, key=lambda value: int(value[1:])):
        entry = grouped[subject]
        if not entry["test_ids"]:
            raise ValueError(f"{subject} has no TEST_SESSION_IDS value in {xlsx_path}")
        if not entry["notes"]:
            raise ValueError(f"{subject} has no training-session note in {xlsx_path}")

        note = entry["notes"][0]
        note_train_ids, note_test_ids = _parse_training_note(note)
        test_ids = sorted(entry["test_ids"])

        if note_test_ids and sorted(note_test_ids) != test_ids:
            warnings.append(
                {
                    "Subject": subject,
                    "warning": "Note TestSession differs from TEST_SESSION_IDS column; using TEST_SESSION_IDS column.",
                    "note_test_ids": ",".join(map(str, note_test_ids)),
                    "column_test_ids": ",".join(map(str, test_ids)),
                    "note": note,
                }
            )

        train_ids = [session_id for session_id in note_train_ids if session_id not in set(test_ids)]
        removed_overlap = [session_id for session_id in note_train_ids if session_id in set(test_ids)]
        if removed_overlap:
            warnings.append(
                {
                    "Subject": subject,
                    "warning": "Removed TEST_SESSION_IDS from training sessions to avoid train/test overlap.",
                    "removed_session_ids": ",".join(map(str, removed_overlap)),
                    "column_test_ids": ",".join(map(str, test_ids)),
                    "note": note,
                }
            )

        plans.append(
            {
                "Subject": subject,
                "INPUT_SENSORS": ",".join(INPUT_SENSORS),
                "TARGET_SENSORS": ",".join(TARGET_SENSORS),
                "TRAIN_SESSION_IDS": train_ids,
                "TEST_SESSION_IDS": test_ids,
                "source_note": note,
            }
        )

    if SUBJECT_FILTER is not None:
        allowed = set(SUBJECT_FILTER)
        plans = [plan for plan in plans if plan["Subject"] in allowed]

    return plans, warnings


def _run_dir() -> Path:
    run_name = RUN_NAME or f"Results_of_MHS_{_sensor_config_name()}_LSTM_TCN_SHRED"
    return _resolve_path(OUTPUT_BASE_DIR) / run_name


def _subject_csv_path(subject: str) -> Path:
    path = NOTEBOOK_DIR / f"{subject}_Gait_xyzShank.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find data file: {path}")
    return path


def _find_config_cell(nb) -> int:
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if (
            "TRAIN_SESSION_IDS" in source
            and "TEST_SESSION_IDS" in source
            and "BEST_MODEL_PATH" in source
        ):
            return idx
    raise ValueError("Could not find the notebook configuration cell.")


def _build_config_source(plan: dict[str, Any], method: str) -> str:
    subject = plan["Subject"]
    sensor_config = _sensor_config_name()
    lines = [
        "# Auto-generated by run_tcn_shred_from_mhs.py.",
        f"Subject = {subject!r}",
        f"SENSOR_CONFIG_NAME = {sensor_config!r}",
        f"CSV_FILE = {str(_subject_csv_path(subject))!r}",
        f"INPUT_SENSORS = {INPUT_SENSORS!r}",
        f"TARGET_SENSORS = {TARGET_SENSORS!r}",
        f"WINDOW_SIZE = {WINDOW_SIZE!r}",
        f"BATCH_SIZE = {BATCH_SIZE!r}",
        "",
        "# Train/test sessions parsed from Results_of_MHS.xlsx.",
        f"TRAIN_SESSION_IDS = {plan['TRAIN_SESSION_IDS']!r}",
        f"TEST_SESSION_IDS = {plan['TEST_SESSION_IDS']!r}",
        f"VALIDATION_SIZE = {VALIDATION_SIZE!r}",
        "",
        f"TRAIN_STRIDE = {TRAIN_STRIDE!r}",
        f"VAL_STRIDE = {VAL_STRIDE!r}",
        f"TEST_STRIDE = {TEST_STRIDE!r}",
        f"EPOCHS = {EPOCHS!r}",
        f"LR = {LR!r}",
        f"BEST_MODEL_PATH = f'{{Subject}}_{{SENSOR_CONFIG_NAME}}_best_gait_{method}.pth'",
        f"TEST_WINDOW_IDX = {TEST_WINDOW_IDX!r}",
        "",
    ]

    if method == "TCN":
        lines.extend(
            [
                f"TCN_CHANNELS = {TCN_CHANNELS!r}",
                f"TCN_KERNEL_SIZE = {TCN_KERNEL_SIZE!r}",
                f"TCN_DROPOUT = {TCN_DROPOUT!r}",
                "",
            ]
        )
    elif method == "SHRED":
        lines.extend(
            [
                "# SHRED hyperparameters from the treadmill-running paper / pyshred defaults.",
                f"SHRED_HIDDEN_SIZE = {SHRED_HIDDEN_SIZE!r}",
                f"SHRED_HIDDEN_LAYERS = {SHRED_HIDDEN_LAYERS!r}",
                f"SHRED_DECODER_L1 = {SHRED_DECODER_L1!r}",
                f"SHRED_DECODER_L2 = {SHRED_DECODER_L2!r}",
                f"SHRED_DROPOUT = {SHRED_DROPOUT!r}",
                "",
            ]
        )

    lines.append("available_sessions = list_available_sessions(CSV_FILE)")
    return "\n".join(lines) + "\n"


def _expected_result_paths(method_dir: Path, subject: str) -> tuple[Path, Path]:
    return (
        method_dir / f"{subject}_test_acc_gyro_weighted_metrics.csv",
        method_dir / f"{subject}_test_acc_gyro_weighted_metric_details.csv",
    )


def _execute_notebook(plan: dict[str, Any], method: str, run_dir: Path) -> tuple[Path, Path, Path, bool]:
    subject = plan["Subject"]
    notebook_path = NOTEBOOK_DIR / NOTEBOOK_FILES[method]
    if not notebook_path.exists():
        raise FileNotFoundError(f"Cannot find notebook for {method}: {notebook_path}")

    method_dir = run_dir / subject / _sensor_config_name() / method
    method_dir.mkdir(parents=True, exist_ok=True)
    summary_path, detail_path = _expected_result_paths(method_dir, subject)
    executed_path = method_dir / f"{subject}_{_sensor_config_name()}_{method}_Ses_executed.ipynb"

    if SKIP_COMPLETED and summary_path.exists() and detail_path.exists():
        print(f"[{subject} {method}] Existing CSV found; skipping.")
        return summary_path, detail_path, executed_path, True

    for stale_path in [summary_path, detail_path]:
        if stale_path.exists():
            stale_path.unlink()

    nb = nbformat.read(notebook_path, as_version=4)
    config_idx = _find_config_cell(nb)
    nb.cells[config_idx].source = _build_config_source(plan, method)

    parameterized_path = method_dir / f"{subject}_{_sensor_config_name()}_{method}_Ses_parameterized.ipynb"
    nbformat.write(nb, parameterized_path)

    ep_kwargs = {
        "timeout": CELL_TIMEOUT,
        "allow_errors": False,
    }
    if KERNEL_NAME:
        ep_kwargs["kernel_name"] = KERNEL_NAME

    print(f"\n[{subject} {method}] Running {notebook_path.name}")
    print(f"[{subject} {method}] Train: {plan['TRAIN_SESSION_IDS']} | Test: {plan['TEST_SESSION_IDS']}")
    ep = ExecutePreprocessor(**ep_kwargs)
    try:
        ep.preprocess(nb, {"metadata": {"path": str(method_dir)}})
    finally:
        nbformat.write(nb, executed_path)

    if not summary_path.exists():
        raise FileNotFoundError(f"{subject} {method} did not create summary CSV: {summary_path}")
    if not detail_path.exists():
        raise FileNotFoundError(f"{subject} {method} did not create detail CSV: {detail_path}")

    return summary_path, detail_path, executed_path, False


def _collect_results(
    plan: dict[str, Any],
    method: str,
    summary_path: Path,
    detail_path: Path,
    skipped: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    metadata = {
        "Method": method,
        "Subject": plan["Subject"],
        "Sensor_Config": _sensor_config_name(),
        "Train_Session_IDS": ",".join(map(str, plan["TRAIN_SESSION_IDS"])),
        "Test_Session_IDS": ",".join(map(str, plan["TEST_SESSION_IDS"])),
        "SkippedExisting": skipped,
    }

    for frame in [summary, detail]:
        frame.drop(columns=[column for column in metadata if column in frame.columns], inplace=True)
        for column, value in reversed(metadata.items()):
            frame.insert(0, column, value)

    return summary, detail


def _plan_dataframe(plans: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for plan in plans:
        rows.append(
            {
                "Subject": plan["Subject"],
                "INPUT_SENSORS": ",".join(INPUT_SENSORS),
                "TARGET_SENSORS": ",".join(TARGET_SENSORS),
                "TRAIN_SESSION_IDS": ",".join(map(str, plan["TRAIN_SESSION_IDS"])),
                "TEST_SESSION_IDS": ",".join(map(str, plan["TEST_SESSION_IDS"])),
                "source_note": plan["source_note"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    xlsx_path = _resolve_path(RESULTS_XLSX)
    run_dir = _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    plans, warnings = _parse_mhs_subject_plan(xlsx_path)
    if not plans:
        raise ValueError(f"No subject plans were parsed from {xlsx_path}")

    plan_output = run_dir / "mhs_session_plan.csv"
    warnings_output = run_dir / "mhs_session_plan_warnings.csv"
    sensor_config = _sensor_config_name()
    failures_output = run_dir / f"ses_models_{sensor_config}_failures.csv"
    summary_output = run_dir / f"ses_models_{sensor_config}_weighted_metrics_summary.csv"
    detail_output = run_dir / f"ses_models_{sensor_config}_weighted_metrics_detail.csv"

    _plan_dataframe(plans).to_csv(plan_output, index=False)
    pd.DataFrame(warnings).to_csv(warnings_output, index=False)

    print(f"Parsed {len(plans)} subject plan(s) from: {xlsx_path}")
    print(f"Run directory: {run_dir}")
    print(f"Plan CSV: {plan_output}")
    if warnings:
        print(f"Warnings CSV: {warnings_output}")

    summary_tables = []
    detail_tables = []
    failures = []

    for plan in plans:
        for method in METHODS:
            try:
                summary_path, detail_path, _, skipped = _execute_notebook(plan, method, run_dir)
                summary, detail = _collect_results(plan, method, summary_path, detail_path, skipped)
                summary_tables.append(summary)
                detail_tables.append(detail)
            except Exception as exc:
                failure = {
                    "Subject": plan["Subject"],
                    "Method": method,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "TRAIN_SESSION_IDS": ",".join(map(str, plan["TRAIN_SESSION_IDS"])),
                    "TEST_SESSION_IDS": ",".join(map(str, plan["TEST_SESSION_IDS"])),
                }
                failures.append(failure)
                pd.DataFrame(failures).to_csv(failures_output, index=False)
                print(f"[{plan['Subject']} {method}] FAILED: {type(exc).__name__}: {exc}")
                if not CONTINUE_ON_ERROR:
                    raise

    if summary_tables:
        combined_summary = pd.concat(summary_tables, ignore_index=True)
        combined_summary.to_csv(summary_output, index=False)
        print(f"\nSaved summary CSV: {summary_output}")
        print(combined_summary.to_string(index=False))
    else:
        print("\nNo summary rows were produced.")

    if detail_tables:
        combined_detail = pd.concat(detail_tables, ignore_index=True)
        combined_detail.to_csv(detail_output, index=False)
        print(f"\nSaved detail CSV: {detail_output}")

    if failures:
        pd.DataFrame(failures).to_csv(failures_output, index=False)
        print(f"Saved failure CSV: {failures_output}")


if __name__ == "__main__":
    main()
