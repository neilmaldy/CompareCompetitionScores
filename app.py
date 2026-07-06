import csv
import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from judge_accuracy import LEVEL_POINTS, judge_accuracy

TARGET_EVENTS = {"SRIF", "SRPF", "SRTF"}
TARGET_JUDGE_TYPES = {"Dm", "Dr", "Dp"}
BREAK_JUDGE_TYPE = "T"
BREAK_MARK_SCHEMA = "break"
MISS_JUDGE_TYPES = {"T", "P"}
MISS_MARK_SCHEMA = "miss"

MARK_MAP = {
    "diffL0.5": 0.5,
    "diffL1": 1,
    "diffL2": 2,
    "diffL3": 3,
    "diffL4": 4,
    "diffL5": 5,
    "diffL6": 6,
    "diffL7": 7,
    "diffL8": 8,
}


@dataclass
class ParsedJudgeRow:
    source: str
    event: str
    entry_number: str
    station_id: str | None
    judge_type: str
    judge_id: str
    marks: list[float]
    mark_events: list[tuple[int, float]]
    assignment_code: str | None = None


@dataclass
class ParsedBreakRow:
    source: str
    event: str
    entry_number: str
    station_id: str | None
    judge_type: str
    judge_id: str
    break_timestamps: list[int] | None
    break_events: list[tuple[int, int]] | None
    tally_break_count: int | None
    assignment_code: str | None = None


@dataclass
class ParsedMissRow:
    source: str
    event: str
    entry_number: str
    station_id: str | None
    judge_type: str
    judge_id: str
    miss_timestamps: list[int] | None
    miss_events: list[tuple[int, int]] | None
    tally_miss_count: int | None
    assignment_code: str | None = None


def parse_payload(raw: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value or value.lower() == "null":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _coerce_timestamp(raw_timestamp: Any) -> int:
    if isinstance(raw_timestamp, int):
        return raw_timestamp
    if isinstance(raw_timestamp, float):
        return int(raw_timestamp)
    if isinstance(raw_timestamp, str):
        try:
            return int(float(raw_timestamp))
        except ValueError:
            return 0
    return 0


def extract_marks(payload: dict[str, Any]) -> tuple[list[float] | None, list[tuple[int, float]] | None]:
    mark_sheet = payload.get("MarkSheet")
    if not isinstance(mark_sheet, dict):
        return None, None

    marks = mark_sheet.get("marks")
    if not isinstance(marks, list):
        return None, None

    # Use sequence first, then timestamp as tie-breaker to preserve intended order.
    sorted_marks = sorted(
        marks,
        key=lambda m: (
            m.get("sequence", 0) if isinstance(m, dict) else 0,
            m.get("timestamp", 0) if isinstance(m, dict) else 0,
        ),
    )

    sequence: list[float] = []
    mark_events: list[tuple[int, float]] = []
    previous_was_difficulty_mark = False
    for mark in sorted_marks:
        if not isinstance(mark, dict):
            previous_was_difficulty_mark = False
            continue
        schema = mark.get("schema")
        if schema in MARK_MAP:
            level = MARK_MAP[schema]
            sequence.append(level)
            mark_events.append((_coerce_timestamp(mark.get("timestamp")), level))
            previous_was_difficulty_mark = True
        elif schema == "undo":
            if previous_was_difficulty_mark and sequence:
                sequence.pop()
            if previous_was_difficulty_mark and mark_events:
                mark_events.pop()
            previous_was_difficulty_mark = False
        elif schema == "clear":
            sequence.clear()
            mark_events.clear()
            previous_was_difficulty_mark = False
        else:
            previous_was_difficulty_mark = False

    return sequence, mark_events


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def extract_break_marks(payload: dict[str, Any]) -> tuple[list[int] | None, list[tuple[int, int]] | None]:
    mark_sheet = payload.get("MarkSheet")
    if not isinstance(mark_sheet, dict):
        return None, None

    marks = mark_sheet.get("marks")
    if not isinstance(marks, list):
        return None, None

    sorted_marks = sorted(
        marks,
        key=lambda m: (
            m.get("sequence", 0) if isinstance(m, dict) else 0,
            m.get("timestamp", 0) if isinstance(m, dict) else 0,
        ),
    )

    break_timestamps: list[int] = []
    previous_was_break_mark = False
    for mark in sorted_marks:
        if not isinstance(mark, dict):
            previous_was_break_mark = False
            continue

        schema = mark.get("schema")
        if schema == BREAK_MARK_SCHEMA:
            break_timestamps.append(_coerce_timestamp(mark.get("timestamp")))
            previous_was_break_mark = True
        elif schema == "undo":
            if previous_was_break_mark and break_timestamps:
                break_timestamps.pop()
            previous_was_break_mark = False
        elif schema == "clear":
            break_timestamps.clear()
            previous_was_break_mark = False
        else:
            previous_was_break_mark = False

    break_events = [(ts, idx) for idx, ts in enumerate(break_timestamps, start=1)]
    return break_timestamps, break_events


def extract_tally_break_count(payload: dict[str, Any]) -> int | None:
    tally_sheet = payload.get("TallySheet")
    if not isinstance(tally_sheet, dict):
        return None

    tally = tally_sheet.get("tally")
    if not isinstance(tally, dict):
        return None

    raw_break = tally.get("break")
    if raw_break is None:
        raw_break = tally.get("breaks")
    return _coerce_int(raw_break)


def extract_miss_marks(payload: dict[str, Any]) -> tuple[list[int] | None, list[tuple[int, int]] | None]:
    mark_sheet = payload.get("MarkSheet")
    if not isinstance(mark_sheet, dict):
        return None, None

    marks = mark_sheet.get("marks")
    if not isinstance(marks, list):
        return None, None

    sorted_marks = sorted(
        marks,
        key=lambda m: (
            m.get("sequence", 0) if isinstance(m, dict) else 0,
            m.get("timestamp", 0) if isinstance(m, dict) else 0,
        ),
    )

    miss_timestamps: list[int] = []
    previous_was_miss_mark = False
    for mark in sorted_marks:
        if not isinstance(mark, dict):
            previous_was_miss_mark = False
            continue

        schema = mark.get("schema")
        if schema == MISS_MARK_SCHEMA:
            miss_timestamps.append(_coerce_timestamp(mark.get("timestamp")))
            previous_was_miss_mark = True
        elif schema == "undo":
            if previous_was_miss_mark and miss_timestamps:
                miss_timestamps.pop()
            previous_was_miss_mark = False
        elif schema == "clear":
            miss_timestamps.clear()
            previous_was_miss_mark = False
        else:
            previous_was_miss_mark = False

    miss_events = [(ts, idx) for idx, ts in enumerate(miss_timestamps, start=1)]
    return miss_timestamps, miss_events


def extract_tally_miss_count(payload: dict[str, Any]) -> int | None:
    tally_sheet = payload.get("TallySheet")
    if not isinstance(tally_sheet, dict):
        return None

    tally = tally_sheet.get("tally")
    if not isinstance(tally, dict):
        return None

    raw_miss = tally.get("miss")
    if raw_miss is None:
        raw_miss = tally.get("misses")
    return _coerce_int(raw_miss)


def extract_judge_meta(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    tally_sheet = payload.get("TallySheet")
    if not isinstance(tally_sheet, dict):
        return None, None

    meta = tally_sheet.get("meta")
    if not isinstance(meta, dict):
        return None, None

    judge_id = meta.get("judgeId")
    judge_type = meta.get("judgeTypeId") or meta.get("judgeTypeID")

    if judge_id is None or judge_type is None:
        return None, None

    return str(judge_id), str(judge_type)


def parse_tsv(content: bytes, is_live: bool) -> tuple[list[ParsedJudgeRow], int]:
    decoded = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")

    parsed_rows: list[ParsedJudgeRow] = []
    skipped_rows = 0

    for row in reader:
        event = (row.get("EventDefinitionAbbr") or "").strip()
        if event not in TARGET_EVENTS:
            continue

        payload = parse_payload(row.get("JudgeScoreDataString") or "")
        if payload is None:
            skipped_rows += 1
            continue

        marks, mark_events = extract_marks(payload)
        judge_id, judge_type = extract_judge_meta(payload)

        if marks is None or mark_events is None or judge_id is None or judge_type is None:
            skipped_rows += 1
            continue

        if judge_type not in TARGET_JUDGE_TYPES:
            continue

        entry_number = (row.get("EntryNumber") or "").strip()
        if not entry_number:
            skipped_rows += 1
            continue

        station_id = (row.get("StationID") or "").strip() or None
        assignment_code = None
        if is_live:
            if not station_id:
                skipped_rows += 1
                continue
            assignment_code = f"{station_id}-{judge_id}"

        parsed_rows.append(
            ParsedJudgeRow(
                source="Live" if is_live else "Shadow",
                event=event,
                entry_number=entry_number,
                station_id=station_id,
                judge_type=judge_type,
                judge_id=judge_id,
                marks=marks,
                mark_events=mark_events,
                assignment_code=assignment_code,
            )
        )

    return parsed_rows, skipped_rows


def parse_tsv_breaks(content: bytes, is_live: bool) -> tuple[list[ParsedBreakRow], int]:
    decoded = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")

    parsed_rows: list[ParsedBreakRow] = []
    skipped_rows = 0

    for row in reader:
        payload = parse_payload(row.get("JudgeScoreDataString") or "")
        if payload is None:
            skipped_rows += 1
            continue

        judge_id, judge_type = extract_judge_meta(payload)
        if judge_id is None or judge_type is None:
            skipped_rows += 1
            continue

        if judge_type != BREAK_JUDGE_TYPE:
            continue

        entry_number = (row.get("EntryNumber") or "").strip()
        if not entry_number:
            skipped_rows += 1
            continue

        break_timestamps, break_events = extract_break_marks(payload)
        tally_break_count = extract_tally_break_count(payload)

        event = (row.get("EventDefinitionAbbr") or "").strip()
        station_id = (row.get("StationID") or "").strip() or None
        assignment_code = f"{station_id or 'NA'}-{judge_id}" if is_live else None

        parsed_rows.append(
            ParsedBreakRow(
                source="Live" if is_live else "Shadow",
                event=event,
                entry_number=entry_number,
                station_id=station_id,
                judge_type=judge_type,
                judge_id=judge_id,
                break_timestamps=break_timestamps,
                break_events=break_events,
                tally_break_count=tally_break_count,
                assignment_code=assignment_code,
            )
        )

    return parsed_rows, skipped_rows


def parse_tsv_misses(content: bytes, is_live: bool) -> tuple[list[ParsedMissRow], int]:
    decoded = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")

    parsed_rows: list[ParsedMissRow] = []
    skipped_rows = 0

    for row in reader:
        payload = parse_payload(row.get("JudgeScoreDataString") or "")
        if payload is None:
            skipped_rows += 1
            continue

        judge_id, judge_type = extract_judge_meta(payload)
        if judge_id is None or judge_type is None:
            skipped_rows += 1
            continue

        if judge_type not in MISS_JUDGE_TYPES:
            continue

        entry_number = (row.get("EntryNumber") or "").strip()
        if not entry_number:
            skipped_rows += 1
            continue

        miss_timestamps, miss_events = extract_miss_marks(payload)
        tally_miss_count = extract_tally_miss_count(payload)

        event = (row.get("EventDefinitionAbbr") or "").strip()
        station_id = (row.get("StationID") or "").strip() or None
        assignment_code = f"{station_id or 'NA'}-{judge_id}" if is_live else None

        parsed_rows.append(
            ParsedMissRow(
                source="Live" if is_live else "Shadow",
                event=event,
                entry_number=entry_number,
                station_id=station_id,
                judge_type=judge_type,
                judge_id=judge_id,
                miss_timestamps=miss_timestamps,
                miss_events=miss_events,
                tally_miss_count=tally_miss_count,
                assignment_code=assignment_code,
            )
        )

    return parsed_rows, skipped_rows


def get_tsv_bytes_from_upload(uploaded_file: Any, label: str) -> bytes | None:
    if uploaded_file is None:
        return None

    filename = (uploaded_file.name or "").lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".tsv"):
        return file_bytes

    if filename.endswith(".zip"):
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
                tsv_names = [
                    name for name in zf.namelist() if not name.endswith("/") and name.lower().endswith(".tsv")
                ]
                if not tsv_names:
                    st.error(f"{label} ZIP does not contain any .tsv files.")
                    return None

                chosen_name = sorted(tsv_names)[0]
                return zf.read(chosen_name)
        except zipfile.BadZipFile:
            st.error(f"{label} file is not a valid ZIP archive.")
            return None

    st.error(f"Unsupported {label} file type. Upload a .tsv or .zip file.")
    return None


def build_shadow_reference(shadow_rows: list[ParsedJudgeRow]) -> dict[tuple[str, str, str], ParsedJudgeRow]:
    reference: dict[tuple[str, str, str], ParsedJudgeRow] = {}
    for row in shadow_rows:
        key = (row.entry_number, row.event, row.judge_type)
        if key not in reference:
            reference[key] = row
    return reference


def compare_rows(
    live_rows: list[ParsedJudgeRow],
    shadow_reference: dict[tuple[str, str, str], ParsedJudgeRow],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for live_row in live_rows:
        key = (live_row.entry_number, live_row.event, live_row.judge_type)
        shadow_row = shadow_reference.get(key)
        if shadow_row is None:
            continue

        if not shadow_row.marks:
            continue

        result = judge_accuracy(shadow_row.marks, live_row.marks)
        records.append(
            {
                "EventDefinitionAbbr": live_row.event,
                "EntryNumber": live_row.entry_number,
                "JudgeTypeID": live_row.judge_type,
                "AssignmentCode": live_row.assignment_code,
                "LiveJudgeId": live_row.judge_id,
                "ShadowJudgeId": shadow_row.judge_id,
                "ShadowMarkCount": len(shadow_row.marks),
                "LiveMarkCount": len(live_row.marks),
                "Accuracy": round(result["accuracy"], 4),
                "NormalizedError": round(result["normalized_error"], 6),
                "AlignmentCost": round(result["alignment_cost"], 6),
                "ReferenceTotal": round(result["reference_total"], 6),
                "JudgeTotal": round(result["judge_total"], 6),
                "TotalErrorPct": round(result["total_error_pct"], 4),
                "ExactMatchRate": round(result["exact_match_rate"], 4),
            }
        )

    return pd.DataFrame(records)


def build_ranking(details_df: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        return pd.DataFrame(
            columns=[
                "Rank",
                "AssignmentCode",
                "AverageAccuracy",
                "Comparisons",
                "AverageExactMatchRate",
                "AverageTotalErrorPct",
            ]
        )

    ranking = (
        details_df.groupby("AssignmentCode", dropna=False)
        .agg(
            AverageAccuracy=("Accuracy", "mean"),
            Comparisons=("Accuracy", "count"),
            AverageExactMatchRate=("ExactMatchRate", "mean"),
            AverageTotalErrorPct=("TotalErrorPct", "mean"),
        )
        .sort_values(by=["AverageAccuracy", "Comparisons"], ascending=[False, False])
        .reset_index()
    )

    ranking["AverageAccuracy"] = ranking["AverageAccuracy"].round(4)
    ranking["AverageExactMatchRate"] = ranking["AverageExactMatchRate"].round(4)
    ranking["AverageTotalErrorPct"] = ranking["AverageTotalErrorPct"].round(4)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    return ranking


def build_excel(details_df: pd.DataFrame, ranking_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        details_df.to_excel(writer, index=False, sheet_name="Judge Accuracy")
        ranking_df.to_excel(writer, index=False, sheet_name="Judge Ranking")
    buffer.seek(0)
    return buffer.read()


def build_full_excel(
    difficulty_details_df: pd.DataFrame,
    difficulty_ranking_df: pd.DataFrame,
    misses_details_df: pd.DataFrame,
    misses_ranking_df: pd.DataFrame,
    breaks_details_df: pd.DataFrame,
    breaks_ranking_df: pd.DataFrame,
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        difficulty_details_df.to_excel(writer, index=False, sheet_name="Difficulty Details")
        difficulty_ranking_df.to_excel(writer, index=False, sheet_name="Difficulty Ranking")
        misses_details_df.to_excel(writer, index=False, sheet_name="Misses Details")
        misses_ranking_df.to_excel(writer, index=False, sheet_name="Misses Ranking")
        breaks_details_df.to_excel(writer, index=False, sheet_name="Breaks Details")
        breaks_ranking_df.to_excel(writer, index=False, sheet_name="Breaks Ranking")
    buffer.seek(0)
    return buffer.read()


def calculate_difficulty_score(mark_levels: list[float]) -> float:
    return sum(LEVEL_POINTS[level] for level in mark_levels if level in LEVEL_POINTS)


def ijru_average(scores: list[float]) -> float | None:
    if not scores:
        return None
    if len(scores) == 1:
        return scores[0]
    if len(scores) == 2:
        return sum(scores) / 2
    if len(scores) == 3:
        ordered = sorted(scores)
        low, mid, high = ordered
        low_gap = mid - low
        high_gap = high - mid
        if low_gap < high_gap:
            return (low + mid) / 2
        # In a tie, athlete benefit applies: average the two higher scores.
        return (mid + high) / 2

    ordered = sorted(scores)
    trimmed = ordered[1:-1]
    if not trimmed:
        return None
    return sum(trimmed) / len(trimmed)


def build_time_series_df(rows: list[ParsedJudgeRow]) -> pd.DataFrame:
    points: list[dict[str, Any]] = []

    for row in rows:
        if not row.mark_events:
            continue

        first_ts = min(ts for ts, _ in row.mark_events)
        label_suffix = row.assignment_code if row.source == "Live" else f"S-{row.judge_id}"
        judge_label = f"{row.source} {label_suffix}"

        for idx, (ts, level) in enumerate(row.mark_events, start=1):
            seconds_since_first = (ts - first_ts) / 1000.0
            points.append(
                {
                    "EntryNumber": row.entry_number,
                    "JudgeTypeID": row.judge_type,
                    "JudgeLabel": judge_label,
                    "Source": row.source,
                    "MarkIndex": idx,
                    "SecondsSinceFirstMark": round(max(0.0, seconds_since_first), 3),
                    "DifficultyLevel": level,
                }
            )

    return pd.DataFrame(points)


def build_score_table(rows: list[ParsedJudgeRow], reference_score: float | None) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    for row in rows:
        calculated = calculate_difficulty_score(row.marks)
        judge_label = row.assignment_code if row.source == "Live" else f"Shadow-{row.judge_id}"

        pct_diff = None
        if reference_score is not None and reference_score != 0:
            pct_diff = (calculated - reference_score) / reference_score * 100.0

        records.append(
            {
                "Source": row.source,
                "Judge": judge_label,
                "JudgeID": row.judge_id,
                "CalculatedDifficultyScore": round(calculated, 4),
                "MarkCount": len(row.marks),
                "ReferenceScore": round(reference_score, 4) if reference_score is not None else None,
                "PercentDifferenceVsReference": round(pct_diff, 4) if pct_diff is not None else None,
            }
        )

    return pd.DataFrame(records)


def first_valid_row(rows: list[ParsedJudgeRow]) -> ParsedJudgeRow | None:
    for row in rows:
        if row.marks:
            return row
    return None


def get_break_count_from_marks(row: ParsedBreakRow) -> int | None:
    if row.break_timestamps is None:
        return None
    return len(row.break_timestamps)


def get_break_count_for_accuracy(row: ParsedBreakRow) -> tuple[int | None, str | None]:
    mark_breaks = get_break_count_from_marks(row)
    if mark_breaks is not None:
        return mark_breaks, "MarkSheet.marks"
    if row.tally_break_count is not None:
        return row.tally_break_count, "TallySheet.tally.break"
    return None, None


def break_accuracy(live_breaks: int, reference_breaks: int) -> float:
    raw = 1.0 - (abs(live_breaks - reference_breaks) / max(reference_breaks, 1))
    return max(0.0, min(1.0, raw))


def build_shadow_break_reference(shadow_rows: list[ParsedBreakRow]) -> dict[str, ParsedBreakRow]:
    reference: dict[str, ParsedBreakRow] = {}
    for row in shadow_rows:
        if row.entry_number not in reference:
            reference[row.entry_number] = row
    return reference


def build_break_time_series_df(rows: list[ParsedBreakRow]) -> pd.DataFrame:
    points: list[dict[str, Any]] = []

    for row in rows:
        label_suffix = row.assignment_code if row.source == "Live" else f"S-{row.judge_id}"
        judge_label = f"{row.source} {label_suffix}"

        if not row.break_events:
            break_count, _ = get_break_count_for_accuracy(row)
            if break_count == 0:
                points.append(
                    {
                        "EntryNumber": row.entry_number,
                        "JudgeLabel": judge_label,
                        "Source": row.source,
                        "SecondsSinceFirstBreak": 0.0,
                        "BreakCount": 0,
                    }
                )
            continue

        first_ts = min(ts for ts, _ in row.break_events)

        for ts, cumulative_breaks in row.break_events:
            seconds_since_first = (ts - first_ts) / 1000.0
            points.append(
                {
                    "EntryNumber": row.entry_number,
                    "JudgeLabel": judge_label,
                    "Source": row.source,
                    "SecondsSinceFirstBreak": round(max(0.0, seconds_since_first), 3),
                    "BreakCount": cumulative_breaks,
                }
            )

    return pd.DataFrame(points)


def dedupe_live_break_rows(rows: list[ParsedBreakRow]) -> list[ParsedBreakRow]:
    deduped: list[ParsedBreakRow] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        assignment = row.assignment_code or f"NA-{row.judge_id}"
        key = (row.entry_number, assignment)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def is_entry_all_zero_breaks(live_rows: list[ParsedBreakRow], shadow_rows: list[ParsedBreakRow]) -> bool:
    combined = live_rows + shadow_rows
    if not combined:
        return False

    for row in combined:
        break_count, _ = get_break_count_for_accuracy(row)
        if break_count is None or break_count != 0:
            return False

    return True


def build_break_accuracy_details(
    live_rows: list[ParsedBreakRow],
    shadow_reference: dict[str, ParsedBreakRow],
    shadow_rows: list[ParsedBreakRow],
) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, Any]] = []
    skipped_no_reference = 0
    deduped_live_rows = dedupe_live_break_rows(live_rows)

    live_by_entry: dict[str, list[ParsedBreakRow]] = {}
    for row in deduped_live_rows:
        live_by_entry.setdefault(row.entry_number, []).append(row)

    shadow_by_entry: dict[str, list[ParsedBreakRow]] = {}
    for row in shadow_rows:
        shadow_by_entry.setdefault(row.entry_number, []).append(row)

    all_zero_entry_cache: dict[str, bool] = {}

    for live_row in deduped_live_rows:
        entry_number = live_row.entry_number

        if entry_number not in all_zero_entry_cache:
            all_zero_entry_cache[entry_number] = is_entry_all_zero_breaks(
                live_by_entry.get(entry_number, []),
                shadow_by_entry.get(entry_number, []),
            )

        all_zero_entry = all_zero_entry_cache[entry_number]
        shadow_row = shadow_reference.get(live_row.entry_number)
        if shadow_row is None:
            skipped_no_reference += 1
            continue

        reference_breaks, _ = get_break_count_for_accuracy(shadow_row)
        if reference_breaks is None:
            if all_zero_entry:
                reference_breaks = 0
            else:
                skipped_no_reference += 1
                continue

        live_breaks, _ = get_break_count_for_accuracy(live_row)
        if live_breaks is None:
            if all_zero_entry:
                live_breaks = 0
            else:
                continue

        accuracy = break_accuracy(live_breaks, reference_breaks)
        percent_difference = (live_breaks - reference_breaks) / max(reference_breaks, 1) * 100.0
        records.append(
            {
                "EntryNumber": live_row.entry_number,
                "AssignmentCode": live_row.assignment_code,
                "LiveJudgeId": live_row.judge_id,
                "ShadowJudgeId": shadow_row.judge_id,
                "LiveBreakCount": live_breaks,
                "ReferenceBreakCount": reference_breaks,
                "Accuracy": round(accuracy, 4),
                "PercentDifferenceVsReference": round(percent_difference, 2),
            }
        )

    return pd.DataFrame(records), skipped_no_reference


def build_break_ranking(details_df: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        return pd.DataFrame(columns=["Rank", "AssignmentCode", "AverageAccuracy", "EntriesCompared"])

    ranking = (
        details_df.groupby("AssignmentCode", dropna=False)
        .agg(
            AverageAccuracy=("Accuracy", "mean"),
            EntriesCompared=("Accuracy", "count"),
        )
        .sort_values(by=["AverageAccuracy", "EntriesCompared"], ascending=[False, False])
        .reset_index()
    )
    ranking["AverageAccuracy"] = ranking["AverageAccuracy"].round(4)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    return ranking


def get_miss_count_from_marks(row: ParsedMissRow) -> int | None:
    if row.miss_timestamps is None:
        return None
    return len(row.miss_timestamps)


def get_miss_count_for_accuracy(row: ParsedMissRow) -> tuple[int | None, str | None]:
    mark_misses = get_miss_count_from_marks(row)
    if mark_misses is not None:
        return mark_misses, "MarkSheet.marks"
    if row.tally_miss_count is not None:
        return row.tally_miss_count, "TallySheet.tally.miss"
    return None, None


def miss_accuracy(live_misses: int, reference_misses: int) -> float:
    raw = 1.0 - (abs(live_misses - reference_misses) / max(reference_misses, 1))
    return max(0.0, min(1.0, raw))


def build_shadow_miss_reference(shadow_rows: list[ParsedMissRow]) -> dict[tuple[str, str], ParsedMissRow]:
    reference: dict[tuple[str, str], ParsedMissRow] = {}
    for row in shadow_rows:
        key = (row.entry_number, row.judge_type)
        if key not in reference:
            reference[key] = row
    return reference


def build_shadow_miss_graph_reference(shadow_rows: list[ParsedMissRow]) -> dict[tuple[str, str], ParsedMissRow]:
    reference: dict[tuple[str, str], ParsedMissRow] = {}
    for row in shadow_rows:
        key = (row.entry_number, row.judge_type)
        if key in reference:
            continue
        if row.miss_events is None:
            continue
        reference[key] = row
    return reference


def build_shadow_miss_accuracy_reference(shadow_rows: list[ParsedMissRow]) -> dict[str, ParsedMissRow]:
    reference: dict[str, ParsedMissRow] = {}
    for row in shadow_rows:
        key = row.entry_number
        if key in reference:
            continue
        if row.tally_miss_count is None:
            continue
        reference[key] = row
    return reference


def build_miss_time_series_df(rows: list[ParsedMissRow]) -> pd.DataFrame:
    points: list[dict[str, Any]] = []

    for row in rows:
        label_suffix = row.assignment_code if row.source == "Live" else f"S-{row.judge_id}"
        judge_label = f"{row.source} {row.judge_type} {label_suffix}"

        if not row.miss_events:
            miss_count, _ = get_miss_count_for_accuracy(row)
            if miss_count == 0:
                points.append(
                    {
                        "EntryNumber": row.entry_number,
                        "JudgeLabel": judge_label,
                        "Source": row.source,
                        "SecondsSinceFirstMiss": 0.0,
                        "MissCount": 0,
                    }
                )
            continue

        first_ts = min(ts for ts, _ in row.miss_events)
        for ts, cumulative_misses in row.miss_events:
            seconds_since_first = (ts - first_ts) / 1000.0
            points.append(
                {
                    "EntryNumber": row.entry_number,
                    "JudgeLabel": judge_label,
                    "Source": row.source,
                    "SecondsSinceFirstMiss": round(max(0.0, seconds_since_first), 3),
                    "MissCount": cumulative_misses,
                }
            )

    return pd.DataFrame(points)


def dedupe_live_miss_rows(rows: list[ParsedMissRow]) -> list[ParsedMissRow]:
    deduped: list[ParsedMissRow] = []
    seen: set[tuple[str, str, str]] = set()

    for row in rows:
        assignment = row.assignment_code or f"NA-{row.judge_id}"
        key = (row.entry_number, row.judge_type, assignment)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def is_entry_judge_type_all_zero_misses(
    live_rows: list[ParsedMissRow],
    shadow_rows: list[ParsedMissRow],
) -> bool:
    combined = live_rows + shadow_rows
    if not combined:
        return False

    for row in combined:
        miss_count, _ = get_miss_count_for_accuracy(row)
        if miss_count is None or miss_count != 0:
            return False

    return True


def build_miss_accuracy_details(
    live_rows: list[ParsedMissRow],
    shadow_reference: dict[str, ParsedMissRow],
    shadow_rows: list[ParsedMissRow],
) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, Any]] = []
    skipped_no_reference = 0
    deduped_live_rows = dedupe_live_miss_rows(live_rows)

    live_by_entry: dict[str, list[ParsedMissRow]] = {}
    for row in deduped_live_rows:
        live_by_entry.setdefault(row.entry_number, []).append(row)

    shadow_by_entry: dict[str, list[ParsedMissRow]] = {}
    for row in shadow_rows:
        shadow_by_entry.setdefault(row.entry_number, []).append(row)

    all_zero_entry_cache: dict[str, bool] = {}

    for live_row in deduped_live_rows:
        entry_number = live_row.entry_number

        if entry_number not in all_zero_entry_cache:
            all_zero_entry_cache[entry_number] = is_entry_judge_type_all_zero_misses(
                live_by_entry.get(entry_number, []),
                shadow_by_entry.get(entry_number, []),
            )

        all_zero_key = all_zero_entry_cache[entry_number]
        shadow_row = shadow_reference.get(entry_number)
        if shadow_row is None:
            skipped_no_reference += 1
            continue

        reference_misses = shadow_row.tally_miss_count
        if reference_misses is None:
            if all_zero_key:
                reference_misses = 0
            else:
                skipped_no_reference += 1
                continue

        live_misses, _ = get_miss_count_for_accuracy(live_row)
        if live_misses is None:
            if all_zero_key:
                live_misses = 0
            else:
                continue

        accuracy = miss_accuracy(live_misses, reference_misses)
        percent_difference = (live_misses - reference_misses) / max(reference_misses, 1) * 100.0
        records.append(
            {
                "EntryNumber": live_row.entry_number,
                "JudgeTypeID": live_row.judge_type,
                "AssignmentCode": live_row.assignment_code,
                "LiveJudgeId": live_row.judge_id,
                "ShadowJudgeId": shadow_row.judge_id,
                "LiveMissCount": live_misses,
                "ReferenceMissCount": reference_misses,
                "Accuracy": round(accuracy, 4),
                "PercentDifferenceVsReference": round(percent_difference, 2),
            }
        )

    return pd.DataFrame(records), skipped_no_reference


def build_miss_ranking(details_df: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        return pd.DataFrame(columns=["Rank", "AssignmentCode", "AverageAccuracy", "EntriesCompared"])

    ranking = (
        details_df.groupby("AssignmentCode", dropna=False)
        .agg(
            AverageAccuracy=("Accuracy", "mean"),
            EntriesCompared=("Accuracy", "count"),
        )
        .sort_values(by=["AverageAccuracy", "EntriesCompared"], ascending=[False, False])
        .reset_index()
    )
    ranking["AverageAccuracy"] = ranking["AverageAccuracy"].round(4)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    return ranking


def build_full_report_excel_from_analysis(analysis_data: dict[str, Any]) -> bytes:
    live_rows = analysis_data.get("live_rows", [])
    shadow_rows = analysis_data.get("shadow_rows", [])
    live_miss_rows = analysis_data.get("live_miss_rows", [])
    shadow_miss_rows = analysis_data.get("shadow_miss_rows", [])
    live_break_rows = analysis_data.get("live_break_rows", [])
    shadow_break_rows = analysis_data.get("shadow_break_rows", [])

    difficulty_reference = build_shadow_reference(shadow_rows)
    difficulty_details_df = compare_rows(live_rows, difficulty_reference)
    difficulty_ranking_df = build_ranking(difficulty_details_df)

    miss_reference = build_shadow_miss_accuracy_reference(shadow_miss_rows)
    misses_details_df, _ = build_miss_accuracy_details(live_miss_rows, miss_reference, shadow_miss_rows)
    misses_ranking_df = build_miss_ranking(misses_details_df)

    if shadow_break_rows:
        break_reference = build_shadow_break_reference(shadow_break_rows)
        breaks_details_df, _ = build_break_accuracy_details(live_break_rows, break_reference, shadow_break_rows)
        breaks_ranking_df = build_break_ranking(breaks_details_df)
    else:
        breaks_details_df = pd.DataFrame(
            columns=[
                "EntryNumber",
                "AssignmentCode",
                "LiveJudgeId",
                "ShadowJudgeId",
                "LiveBreakCount",
                "ReferenceBreakCount",
                "Accuracy",
                "PercentDifferenceVsReference",
            ]
        )
        breaks_ranking_df = pd.DataFrame(columns=["Rank", "AssignmentCode", "AverageAccuracy", "EntriesCompared"])

    return build_full_excel(
        difficulty_details_df,
        difficulty_ranking_df,
        misses_details_df,
        misses_ranking_df,
        breaks_details_df,
        breaks_ranking_df,
    )


def main() -> None:
    st.set_page_config(page_title="AMJRF Judge Accuracy", layout="wide")
    st.title("AMJRF Judge Accuracy Check")

    st.markdown(
        """
Compare live competition judge mark sequences against the first valid matching
shadow judge sequence per EntryNumber, EventDefinitionAbbr, and JudgeTypeID.

Scope:
- Events: SRIF, SRPF, SRTF
- Judge types: Dm, Dr, Dp
- Matching key: (EntryNumber, EventDefinitionAbbr, JudgeTypeID)
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        live_file = st.file_uploader("Live TSV or ZIP", type=["tsv", "zip"], key="live")
    with col2:
        shadow_file = st.file_uploader("Shadow TSV or ZIP", type=["tsv", "zip"], key="shadow")

    if st.button("Run Analysis", type="primary"):
        if not live_file:
            st.info("Upload a live TSV/ZIP to begin. Upload shadow TSV/ZIP for side-by-side comparison.")
        else:
            live_bytes = get_tsv_bytes_from_upload(live_file, "Live")
            shadow_bytes = get_tsv_bytes_from_upload(shadow_file, "Shadow") if shadow_file else None

            if live_bytes is not None:
                live_rows, live_skipped = parse_tsv(live_bytes, is_live=True)
                live_break_rows, live_break_skipped = parse_tsv_breaks(live_bytes, is_live=True)
                live_miss_rows, live_miss_skipped = parse_tsv_misses(live_bytes, is_live=True)
                shadow_rows: list[ParsedJudgeRow] = []
                shadow_skipped = 0
                shadow_break_rows: list[ParsedBreakRow] = []
                shadow_break_skipped = 0
                shadow_miss_rows: list[ParsedMissRow] = []
                shadow_miss_skipped = 0
                if shadow_bytes is not None:
                    shadow_rows, shadow_skipped = parse_tsv(shadow_bytes, is_live=False)
                    shadow_break_rows, shadow_break_skipped = parse_tsv_breaks(shadow_bytes, is_live=False)
                    shadow_miss_rows, shadow_miss_skipped = parse_tsv_misses(shadow_bytes, is_live=False)

                st.session_state["analysis_data"] = {
                    "live_rows": live_rows,
                    "shadow_rows": shadow_rows,
                    "live_skipped": live_skipped,
                    "shadow_skipped": shadow_skipped,
                    "live_break_rows": live_break_rows,
                    "shadow_break_rows": shadow_break_rows,
                    "live_break_skipped": live_break_skipped,
                    "shadow_break_skipped": shadow_break_skipped,
                    "live_miss_rows": live_miss_rows,
                    "shadow_miss_rows": shadow_miss_rows,
                    "live_miss_skipped": live_miss_skipped,
                    "shadow_miss_skipped": shadow_miss_skipped,
                }

    difficulty_tab, misses_tab, breaks_tab = st.tabs([
        "Difficulty Judge Analysis",
        "Misses",
        "Breaks",
    ])

    with misses_tab:
        if not live_file:
            st.info("Upload a live TSV/ZIP to begin. Upload shadow TSV/ZIP for side-by-side comparison.")
            return

        analysis_data = st.session_state.get("analysis_data")
        if analysis_data is None:
            st.info("Click 'Run Analysis' to parse uploaded files.")
            return

        live_miss_rows = analysis_data.get("live_miss_rows", [])
        shadow_miss_rows = analysis_data.get("shadow_miss_rows", [])
        live_miss_skipped = analysis_data.get("live_miss_skipped", 0)
        shadow_miss_skipped = analysis_data.get("shadow_miss_skipped", 0)

        st.subheader("Misses Summary")
        m1, m2 = st.columns(2)
        m1.metric("Parsed Live Miss Rows (T/P)", len(live_miss_rows))
        m2.metric("Parsed Shadow Miss Rows (T/P)", len(shadow_miss_rows))
        st.caption(f"Skipped rows during miss parse: live={live_miss_skipped}, shadow={shadow_miss_skipped}")

        shadow_miss_graph_reference = build_shadow_miss_graph_reference(shadow_miss_rows)
        shadow_miss_accuracy_reference = build_shadow_miss_accuracy_reference(shadow_miss_rows)
        shadow_entries_with_ref_data = {
            entry
            for entry, row in shadow_miss_accuracy_reference.items()
            if row.tally_miss_count is not None
        }

        station_options = sorted({row.station_id for row in live_miss_rows if row.station_id})
        selected_station = st.selectbox("StationID Filter", ["All"] + station_options, key="misses_station")
        limit_to_shadow_miss_data = st.checkbox(
            "Only include entries with miss data in shadow competition",
            value=False,
            key="misses_shadow_only",
        )

        filtered_live_miss_rows = live_miss_rows
        if selected_station != "All":
            filtered_live_miss_rows = [row for row in filtered_live_miss_rows if row.station_id == selected_station]

        if limit_to_shadow_miss_data:
            filtered_live_miss_rows = [
                row for row in filtered_live_miss_rows if row.entry_number in shadow_entries_with_ref_data
            ]

        live_entries = sorted(
            {row.entry_number for row in filtered_live_miss_rows},
            key=lambda x: int(x) if x.isdigit() else x,
        )

        if not live_entries:
            if limit_to_shadow_miss_data:
                st.warning("No live entries for this StationID have miss reference data in shadow competition.")
            else:
                st.warning("No live miss-judge entries are available for the selected StationID.")
        else:
            selected_entry = st.selectbox("Entry Number", live_entries, key="misses_entry")
            graph_source = st.selectbox(
                "Graph Score Sets",
                ["Both live and shadow", "Live only", "Shadow only"],
                key="misses_graph_source",
            )

            entry_live_rows = [row for row in filtered_live_miss_rows if row.entry_number == selected_entry]
            entry_shadow_rows = [
                row
                for (entry_number, _judge_type), row in shadow_miss_graph_reference.items()
                if entry_number == selected_entry
            ]
            if graph_source == "Live only":
                entry_rows_for_chart = entry_live_rows
            elif graph_source == "Shadow only":
                entry_rows_for_chart = entry_shadow_rows
            else:
                entry_rows_for_chart = entry_live_rows + entry_shadow_rows

            chart_df = build_miss_time_series_df(entry_rows_for_chart)
            if chart_df.empty:
                st.warning("No timestamped miss marks found for this entry.")
            else:
                st.caption("X-axis is seconds since each judge's first miss mark.")
                marker_symbols = [
                    "circle",
                    "square",
                    "triangle",
                    "diamond",
                    "cross",
                    "star",
                    "triangle-up",
                    "triangle-down",
                    "wedge",
                    "arrow",
                ]
                judge_labels = sorted(chart_df["JudgeLabel"].unique().tolist())
                shape_map = {
                    label: marker_symbols[idx % len(marker_symbols)]
                    for idx, label in enumerate(judge_labels)
                }

                base = alt.Chart(chart_df).encode(
                    x=alt.X("SecondsSinceFirstMiss:Q", title="Seconds Since First Miss"),
                    y=alt.Y("MissCount:Q", title="Cumulative Miss Count"),
                    color=alt.Color("JudgeLabel:N", title="Score Set"),
                )

                dashed_trend = base.mark_line(strokeDash=[4, 4], opacity=0.28)
                points = base.mark_point(size=90, filled=True).encode(
                    shape=alt.Shape(
                        "JudgeLabel:N",
                        title="Score Set",
                        scale=alt.Scale(domain=judge_labels, range=[shape_map[label] for label in judge_labels]),
                    ),
                    tooltip=[
                        alt.Tooltip("JudgeLabel:N", title="Score Set"),
                        alt.Tooltip("Source:N", title="Source"),
                        alt.Tooltip("SecondsSinceFirstMiss:Q", title="Seconds", format=".2f"),
                        alt.Tooltip("MissCount:Q", title="Miss Count"),
                    ],
                )

                miss_plot = alt.layer(dashed_trend, points).properties(height=360)
                st.altair_chart(miss_plot, width="stretch")

        miss_details_df, skipped_no_reference = build_miss_accuracy_details(
            filtered_live_miss_rows,
            shadow_miss_accuracy_reference,
            shadow_miss_rows,
        )
        miss_ranking_df = build_miss_ranking(miss_details_df)

        st.caption("Per-entry accuracy formula: 1 - abs(live - reference) / max(reference, 1), clamped to [0, 1].")
        st.caption(
            f"Entries skipped for miss accuracy due to missing shadow reference miss count: {skipped_no_reference}"
        )

        st.subheader("Miss Judge Accuracy Details")
        if miss_details_df.empty:
            st.info("No comparable miss rows found for the current filter.")
        else:
            display_miss_details_df = miss_details_df.drop(columns=["Accuracy"], errors="ignore")
            st.dataframe(display_miss_details_df, width="stretch")

        st.subheader("Miss Judge Accuracy Ranking")
        if miss_ranking_df.empty:
            st.info("No miss-judge ranking is available for the current filter.")
        else:
            st.dataframe(miss_ranking_df, width="stretch")

        full_report_bytes = build_full_report_excel_from_analysis(analysis_data)
        st.download_button(
            label="Download Full Excel Report",
            data=full_report_bytes,
            file_name="amjrf_judge_full_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_full_report_misses",
        )

    with difficulty_tab:
        if not live_file:
            st.info("Upload a live TSV/ZIP to begin. Upload shadow TSV/ZIP for side-by-side comparison.")
            return

        analysis_data = st.session_state.get("analysis_data")
        if analysis_data is None:
            st.info("Click 'Run Analysis' to parse uploaded files.")
            return

        live_rows = analysis_data["live_rows"]
        shadow_rows = analysis_data["shadow_rows"]
        live_skipped = analysis_data["live_skipped"]
        shadow_skipped = analysis_data["shadow_skipped"]

        shadow_reference = build_shadow_reference(shadow_rows)
        details_df = compare_rows(live_rows, shadow_reference)
        ranking_df = build_ranking(details_df)

        st.subheader("Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Parsed Live Rows", len(live_rows))
        s2.metric("Parsed Shadow Rows", len(shadow_rows))
        s3.metric("Compared Rows", len(details_df))
        s4.metric("Unique Ranked Judges", len(ranking_df))

        st.caption(f"Skipped rows during parse: live={live_skipped}, shadow={shadow_skipped}")

        st.subheader("Difficulty Timeline Explorer")
        explorer_rows = [row for row in (live_rows + shadow_rows) if row.judge_type in TARGET_JUDGE_TYPES]
        if not explorer_rows:
            st.warning("No D-type judge data found for the selected event filter.")
        else:
            live_explorer_rows = [row for row in explorer_rows if row.source == "Live"]
            shadow_explorer_rows = [row for row in explorer_rows if row.source == "Shadow"]

            station_options = sorted({row.station_id for row in live_explorer_rows if row.station_id})
            selected_station = st.selectbox("StationID Filter", ["All"] + station_options)
            limit_to_shadow_entries = st.checkbox(
                "Only include entries found in shadow competition",
                value=False,
            )

            station_filtered_live_rows = live_explorer_rows
            if selected_station != "All":
                station_filtered_live_rows = [row for row in live_explorer_rows if row.station_id == selected_station]

            shadow_entries = {row.entry_number for row in shadow_explorer_rows}
            if limit_to_shadow_entries:
                station_filtered_live_rows = [
                    row for row in station_filtered_live_rows if row.entry_number in shadow_entries
                ]

            selected_live_entries = {row.entry_number for row in station_filtered_live_rows}
            station_filtered_shadow_rows = [
                row for row in shadow_explorer_rows if row.entry_number in selected_live_entries
            ]

            station_filtered_rows = station_filtered_live_rows + station_filtered_shadow_rows

            entry_options = sorted(
                selected_live_entries,
                key=lambda x: int(x) if x.isdigit() else x,
            )

            if not entry_options:
                if limit_to_shadow_entries:
                    st.warning("No live entries for this StationID were found in the shadow competition.")
                else:
                    st.warning("No live entries are available for the selected StationID.")
            else:
                selected_entry = st.selectbox("Entry Number", entry_options)
                entry_rows = [row for row in station_filtered_rows if row.entry_number == selected_entry]

                if not entry_rows:
                    st.warning("No difficulty rows are available for the selected StationID.")
                else:
                    available_types = sorted({row.judge_type for row in entry_rows})
                    valid_type_options = sorted({row.judge_type for row in entry_rows if row.marks})
                    unavailable_types = [jt for jt in available_types if jt not in valid_type_options]

                    if unavailable_types:
                        unavailable_text = ", ".join(f"{jt} (no valid marks)" for jt in unavailable_types)
                        st.info(f"Unavailable judge types for this entry: {unavailable_text}")

                    if not valid_type_options:
                        st.warning("No valid D-type marks are available for this entry.")
                    else:
                        selected_judge_type = st.selectbox("Judge Type", valid_type_options)
                        graph_source = st.selectbox(
                            "Graph Score Sets",
                            ["Both live and shadow", "Live only", "Shadow only"],
                            key="difficulty_graph_source",
                        )
                        selected_rows = [row for row in entry_rows if row.judge_type == selected_judge_type]

                        if graph_source == "Live only":
                            selected_rows = [row for row in selected_rows if row.source == "Live"]
                        elif graph_source == "Shadow only":
                            selected_rows = [row for row in selected_rows if row.source == "Shadow"]

                        timeline_df = build_time_series_df(selected_rows)
                        if timeline_df.empty:
                            st.warning("No timestamped marks found for this entry/judge type.")
                        else:
                            st.caption("X-axis is seconds since each judge's first mark.")
                            marker_symbols = [
                                "circle",
                                "square",
                                "triangle",
                                "diamond",
                                "cross",
                                "star",
                                "triangle-up",
                                "triangle-down",
                                "wedge",
                                "arrow",
                            ]
                            judge_labels = sorted(timeline_df["JudgeLabel"].unique().tolist())
                            shape_map = {
                                label: marker_symbols[idx % len(marker_symbols)]
                                for idx, label in enumerate(judge_labels)
                            }

                            base = alt.Chart(timeline_df).encode(
                                x=alt.X("SecondsSinceFirstMark:Q", title="Seconds Since First Mark"),
                                y=alt.Y("DifficultyLevel:Q", title="Difficulty Level"),
                                color=alt.Color("JudgeLabel:N", title="Score Set"),
                            )

                            dashed_trend = base.mark_line(strokeDash=[4, 4], opacity=0.28)

                            points = base.mark_point(size=90, filled=True).encode(
                                shape=alt.Shape(
                                    "JudgeLabel:N",
                                    title="Score Set",
                                    scale=alt.Scale(
                                        domain=judge_labels,
                                        range=[shape_map[label] for label in judge_labels],
                                    ),
                                ),
                                tooltip=[
                                    alt.Tooltip("JudgeLabel:N", title="Score Set"),
                                    alt.Tooltip("Source:N", title="Source"),
                                    alt.Tooltip("MarkIndex:Q", title="Mark #"),
                                    alt.Tooltip("SecondsSinceFirstMark:Q", title="Seconds", format=".2f"),
                                    alt.Tooltip("DifficultyLevel:Q", title="Difficulty", format=".1f"),
                                ],
                            )

                            dot_plot = alt.layer(dashed_trend, points).properties(height=360)
                            st.altair_chart(dot_plot, width="stretch")

                        live_selected = [row for row in selected_rows if row.source == "Live"]
                        shadow_selected = [row for row in selected_rows if row.source == "Shadow"]

                        shadow_first_match = first_valid_row(shadow_selected)
                        reference_label = ""
                        reference_score: float | None = None

                        if shadow_first_match is not None:
                            reference_score = calculate_difficulty_score(shadow_first_match.marks)
                            reference_label = "Shadow first-match reference score"
                        else:
                            live_scores = [calculate_difficulty_score(row.marks) for row in live_selected if row.marks]
                            reference_score = ijru_average(live_scores)
                            reference_label = "IJRU 4.2 averaged live reference score"

                        if reference_score is not None:
                            st.caption(f"Reference source: {reference_label}")
                            st.metric(reference_label, round(reference_score, 4))
                        else:
                            st.warning("Reference score is unavailable for this selection.")

                        st.subheader("Calculated Judge Scores")
                        score_df = build_score_table(selected_rows, reference_score)
                        if score_df.empty:
                            st.info("No judge scores available for this selection.")
                        else:
                            st.dataframe(score_df, width="stretch")

        if details_df.empty:
            st.warning("No comparable rows found after filtering and matching.")
        else:
            st.subheader("Judge Accuracy Details")
            st.dataframe(details_df, width="stretch")

            st.subheader("Judge Ranking")
            st.dataframe(ranking_df, width="stretch")

            excel_bytes = build_full_report_excel_from_analysis(analysis_data)
            st.download_button(
                label="Download Full Excel Report",
                data=excel_bytes,
                file_name="amjrf_judge_full_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_full_report_difficulty",
            )

    with breaks_tab:
        if not live_file:
            st.info("Upload a live TSV/ZIP to begin. Upload shadow TSV/ZIP for side-by-side comparison.")
            return

        analysis_data = st.session_state.get("analysis_data")
        if analysis_data is None:
            st.info("Click 'Run Analysis' in the Difficulty tab to parse uploaded files.")
            return

        live_break_rows = analysis_data.get("live_break_rows", [])
        shadow_break_rows = analysis_data.get("shadow_break_rows", [])
        live_break_skipped = analysis_data.get("live_break_skipped", 0)
        shadow_break_skipped = analysis_data.get("shadow_break_skipped", 0)
        has_shadow_break_data = len(shadow_break_rows) > 0

        st.subheader("Breaks Summary")
        b1, b2 = st.columns(2)
        b1.metric("Parsed Live Technical Rows", len(live_break_rows))
        b2.metric("Parsed Shadow Technical Rows", len(shadow_break_rows))
        st.caption(
            f"Skipped rows during technical parse: live={live_break_skipped}, shadow={shadow_break_skipped}"
        )

        shadow_break_reference = build_shadow_break_reference(shadow_break_rows)
        shadow_entries_with_ref_data = {
            entry
            for entry, row in shadow_break_reference.items()
            if get_break_count_for_accuracy(row)[0] is not None
        }

        station_options = sorted({row.station_id for row in live_break_rows if row.station_id})
        selected_station = st.selectbox("StationID Filter", ["All"] + station_options, key="breaks_station")
        if has_shadow_break_data:
            limit_to_shadow_break_data = st.checkbox(
                "Only include entries with technical break data in shadow competition",
                value=False,
                key="breaks_shadow_only",
            )
        else:
            limit_to_shadow_break_data = False
            st.info("No shadow competition data provided. Showing live break timeline only.")

        filtered_live_break_rows = live_break_rows
        if selected_station != "All":
            filtered_live_break_rows = [
                row for row in filtered_live_break_rows if row.station_id == selected_station
            ]

        if limit_to_shadow_break_data:
            filtered_live_break_rows = [
                row for row in filtered_live_break_rows if row.entry_number in shadow_entries_with_ref_data
            ]

        live_entries = sorted(
            {row.entry_number for row in filtered_live_break_rows},
            key=lambda x: int(x) if x.isdigit() else x,
        )

        if not live_entries:
            if limit_to_shadow_break_data:
                st.warning(
                    "No live entries for this StationID have technical break reference data in shadow competition."
                )
            else:
                st.warning("No live technical-judge entries are available for the selected StationID.")
        else:
            selected_entry = st.selectbox("Entry Number", live_entries, key="breaks_entry")
            graph_source = st.selectbox(
                "Graph Score Sets",
                ["Both live and shadow", "Live only", "Shadow only"],
                key="breaks_graph_source",
            )

            entry_live_rows = [row for row in filtered_live_break_rows if row.entry_number == selected_entry]
            entry_shadow_rows = [row for row in shadow_break_rows if row.entry_number == selected_entry]
            if graph_source == "Live only":
                entry_rows_for_chart = entry_live_rows
            elif graph_source == "Shadow only":
                entry_rows_for_chart = entry_shadow_rows
            else:
                entry_rows_for_chart = entry_live_rows + entry_shadow_rows

            chart_df = build_break_time_series_df(entry_rows_for_chart)
            if chart_df.empty:
                st.warning("No timestamped break marks found for this entry.")
            else:
                st.caption("X-axis is seconds since each technical judge's first break mark.")
                marker_symbols = [
                    "circle",
                    "square",
                    "triangle",
                    "diamond",
                    "cross",
                    "star",
                    "triangle-up",
                    "triangle-down",
                    "wedge",
                    "arrow",
                ]
                judge_labels = sorted(chart_df["JudgeLabel"].unique().tolist())
                shape_map = {
                    label: marker_symbols[idx % len(marker_symbols)]
                    for idx, label in enumerate(judge_labels)
                }

                base = alt.Chart(chart_df).encode(
                    x=alt.X("SecondsSinceFirstBreak:Q", title="Seconds Since First Break"),
                    y=alt.Y("BreakCount:Q", title="Cumulative Break Count"),
                    color=alt.Color("JudgeLabel:N", title="Score Set"),
                )

                dashed_trend = base.mark_line(strokeDash=[4, 4], opacity=0.28)
                points = base.mark_point(size=90, filled=True).encode(
                    shape=alt.Shape(
                        "JudgeLabel:N",
                        title="Score Set",
                        scale=alt.Scale(domain=judge_labels, range=[shape_map[label] for label in judge_labels]),
                    ),
                    tooltip=[
                        alt.Tooltip("JudgeLabel:N", title="Score Set"),
                        alt.Tooltip("Source:N", title="Source"),
                        alt.Tooltip("SecondsSinceFirstBreak:Q", title="Seconds", format=".2f"),
                        alt.Tooltip("BreakCount:Q", title="Break Count"),
                    ],
                )

                break_plot = alt.layer(dashed_trend, points).properties(height=360)
                st.altair_chart(break_plot, width="stretch")

            if has_shadow_break_data:
                selected_shadow_reference = shadow_break_reference.get(selected_entry)
                selected_reference_breaks = None
                selected_reference_source = "Unavailable"
                if selected_shadow_reference is not None:
                    selected_reference_breaks, selected_reference_source = get_break_count_for_accuracy(
                        selected_shadow_reference
                    )

                if selected_reference_breaks is not None:
                    st.caption(f"Reference source: {selected_reference_source}")
                    st.metric("Shadow reference break count", selected_reference_breaks)
                else:
                    st.info(
                        "Shadow break reference is unavailable for this entry. Accuracy is not calculated for this entry."
                    )

        if has_shadow_break_data:
            break_details_df, skipped_no_reference = build_break_accuracy_details(
                filtered_live_break_rows,
                shadow_break_reference,
                shadow_break_rows,
            )
            break_ranking_df = build_break_ranking(break_details_df)

            st.caption(
                "Per-entry accuracy formula: 1 - abs(live - reference) / max(reference, 1), clamped to [0, 1]."
            )
            st.caption(
                f"Entries skipped for break accuracy due to missing shadow reference break count: {skipped_no_reference}"
            )

            st.subheader("Technical Judge Break Accuracy Details")
            if break_details_df.empty:
                st.info("No comparable technical break rows found for the current filter.")
            else:
                display_break_details_df = break_details_df.drop(columns=["Accuracy"], errors="ignore")
                st.dataframe(display_break_details_df, width="stretch")

            st.subheader("Technical Judge Break Accuracy Ranking")
            if break_ranking_df.empty:
                st.info("No technical-judge ranking is available for the current filter.")
            else:
                st.dataframe(break_ranking_df, width="stretch")
        else:
            st.subheader("Technical Judge Break Accuracy Details")
            st.info("Break accuracy analysis is skipped until shadow competition data is provided.")

            st.subheader("Technical Judge Break Accuracy Ranking")
            st.info("Break ranking is unavailable without shadow competition data.")

        full_report_bytes = build_full_report_excel_from_analysis(analysis_data)
        st.download_button(
            label="Download Full Excel Report",
            data=full_report_bytes,
            file_name="amjrf_judge_full_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_full_report_breaks",
        )


if __name__ == "__main__":
    main()
