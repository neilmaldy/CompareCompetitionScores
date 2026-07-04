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
    for mark in sorted_marks:
        if not isinstance(mark, dict):
            continue
        schema = mark.get("schema")
        if schema in MARK_MAP:
            level = MARK_MAP[schema]
            sequence.append(level)
            mark_events.append((_coerce_timestamp(mark.get("timestamp")), level))
        elif schema == "undo":
            if sequence:
                sequence.pop()
            if mark_events:
                mark_events.pop()
        elif schema == "clear":
            sequence.clear()
            mark_events.clear()

    return sequence, mark_events


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

    if not live_file:
        st.info("Upload a live TSV/ZIP to begin. Upload shadow TSV/ZIP for side-by-side comparison.")
        return

    if st.button("Run Analysis", type="primary"):
        live_bytes = get_tsv_bytes_from_upload(live_file, "Live")
        shadow_bytes = get_tsv_bytes_from_upload(shadow_file, "Shadow") if shadow_file else None

        if live_bytes is None:
            return

        live_rows, live_skipped = parse_tsv(live_bytes, is_live=True)
        shadow_rows: list[ParsedJudgeRow] = []
        shadow_skipped = 0
        if shadow_bytes is not None:
            shadow_rows, shadow_skipped = parse_tsv(shadow_bytes, is_live=False)

        st.session_state["analysis_data"] = {
            "live_rows": live_rows,
            "shadow_rows": shadow_rows,
            "live_skipped": live_skipped,
            "shadow_skipped": shadow_skipped,
        }

    analysis_data = st.session_state.get("analysis_data")
    if analysis_data is None:
        st.info("Click 'Run Analysis' to parse uploaded files.")
        return

    live_rows: list[ParsedJudgeRow] = analysis_data["live_rows"]
    shadow_rows: list[ParsedJudgeRow] = analysis_data["shadow_rows"]
    live_skipped: int = analysis_data["live_skipped"]
    shadow_skipped: int = analysis_data["shadow_skipped"]

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
                    selected_rows = [row for row in entry_rows if row.judge_type == selected_judge_type]

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
                            label: marker_symbols[idx % len(marker_symbols)] for idx, label in enumerate(judge_labels)
                        }
                        timeline_df["MarkerShape"] = timeline_df["JudgeLabel"].map(shape_map)

                        base = alt.Chart(timeline_df).encode(
                            x=alt.X("SecondsSinceFirstMark:Q", title="Seconds Since First Mark"),
                            y=alt.Y("DifficultyLevel:Q", title="Difficulty Level"),
                            color=alt.Color("JudgeLabel:N", title="Score Set"),
                        )

                        dashed_trend = base.mark_line(strokeDash=[4, 4], opacity=0.28)

                        points = base.mark_point(size=90, filled=True).encode(
                            shape=alt.Shape(
                                "MarkerShape:N",
                                title="Marker",
                                scale=alt.Scale(domain=list(shape_map.values()), range=marker_symbols),
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
        return

    st.subheader("Judge Accuracy Details")
    st.dataframe(details_df, width="stretch")

    st.subheader("Judge Ranking")
    st.dataframe(ranking_df, width="stretch")

    excel_bytes = build_excel(details_df, ranking_df)
    st.download_button(
        label="Download Excel Report",
        data=excel_bytes,
        file_name="amjrf_judge_accuracy_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
