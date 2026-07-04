import csv
import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from judge_accuracy import judge_accuracy

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
    event: str
    entry_number: str
    judge_type: str
    judge_id: str
    marks: list[float]
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


def extract_marks(payload: dict[str, Any]) -> list[float] | None:
    mark_sheet = payload.get("MarkSheet")
    if not isinstance(mark_sheet, dict):
        return None

    marks = mark_sheet.get("marks")
    if not isinstance(marks, list):
        return None

    # Use sequence first, then timestamp as tie-breaker to preserve intended order.
    sorted_marks = sorted(
        marks,
        key=lambda m: (
            m.get("sequence", 0) if isinstance(m, dict) else 0,
            m.get("timestamp", 0) if isinstance(m, dict) else 0,
        ),
    )

    sequence: list[float] = []
    for mark in sorted_marks:
        if not isinstance(mark, dict):
            continue
        schema = mark.get("schema")
        if schema in MARK_MAP:
            sequence.append(MARK_MAP[schema])
        elif schema == "undo":
            if sequence:
                sequence.pop()

    return sequence


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

        marks = extract_marks(payload)
        judge_id, judge_type = extract_judge_meta(payload)

        if marks is None or judge_id is None or judge_type is None:
            skipped_rows += 1
            continue

        if judge_type not in TARGET_JUDGE_TYPES:
            continue

        entry_number = (row.get("EntryNumber") or "").strip()
        if not entry_number:
            skipped_rows += 1
            continue

        assignment_code = None
        if is_live:
            station_id = (row.get("StationID") or "").strip()
            if not station_id:
                skipped_rows += 1
                continue
            assignment_code = f"{station_id}-{judge_id}"

        parsed_rows.append(
            ParsedJudgeRow(
                event=event,
                entry_number=entry_number,
                judge_type=judge_type,
                judge_id=judge_id,
                marks=marks,
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

    if not live_file or not shadow_file:
        st.info("Upload both TSV files to run analysis.")
        return

    if st.button("Run Analysis", type="primary"):
        live_bytes = get_tsv_bytes_from_upload(live_file, "Live")
        shadow_bytes = get_tsv_bytes_from_upload(shadow_file, "Shadow")

        if live_bytes is None or shadow_bytes is None:
            return

        live_rows, live_skipped = parse_tsv(live_bytes, is_live=True)
        shadow_rows, shadow_skipped = parse_tsv(shadow_bytes, is_live=False)

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
