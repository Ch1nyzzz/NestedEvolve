"""Streamlit app for the local NOA runtime dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import psutil
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from dashboard.data import (
    ACTIVE_RUN_WINDOW_SEC,
    HEARTBEAT_STALE_SEC,
    build_run_snapshot,
    format_elapsed,
    list_runs,
    load_detail_payload,
    patch_preview_from_ops,
    pick_default_run,
    resolve_detail_file,
)

ROOT = Path(__file__).resolve().parent.parent / ".noa_runs"


def _process_tree_metrics(pid: int | None) -> dict[str, float | int | None]:
    if not pid:
        return {"cpu_percent": None, "rss_mb": None, "process_count": 0}
    try:
        root = psutil.Process(pid)
    except psutil.Error:
        return {"cpu_percent": None, "rss_mb": None, "process_count": 0}

    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass

    cpu = 0.0
    rss = 0
    alive = 0
    for proc in processes:
        try:
            if not proc.is_running():
                continue
            alive += 1
            cpu += proc.cpu_percent(interval=None)
            rss += proc.memory_info().rss
        except psutil.Error:
            continue
    return {
        "cpu_percent": round(cpu, 1),
        "rss_mb": round(rss / (1024 * 1024), 1),
        "process_count": alive,
    }


def _render_overview(snapshot: dict) -> None:
    current_run = snapshot["current_run"]
    metrics = _process_tree_metrics(current_run.get("pid"))
    cols = st.columns(6)
    cols[0].metric("Run", snapshot["run_id"])
    cols[1].metric("Status", str(current_run.get("status", "unknown")))
    cols[2].metric("Layer", str(current_run.get("active_layer") or "-"))
    cols[3].metric("Spawn", str(current_run.get("active_spawn_id") or "-"))
    cols[4].metric("Active Children", len(snapshot["active_children"]))
    cols[5].metric("Stale Children", len(snapshot["stale_children"]))

    cols = st.columns(4)
    cols[0].metric("Elapsed", format_elapsed(current_run.get("started_at")))
    cols[1].metric(
        "CPU %", metrics["cpu_percent"] if metrics["cpu_percent"] is not None else "n/a"
    )
    cols[2].metric(
        "RSS MB", metrics["rss_mb"] if metrics["rss_mb"] is not None else "n/a"
    )
    cols[3].metric("Proc Tree", metrics["process_count"])

    with st.expander("Current Run Metadata", expanded=False):
        st.json(current_run)


def _render_llm(snapshot: dict) -> None:
    summary = snapshot["llm_summary"]
    current_run = snapshot["current_run"]
    cols = st.columns(7)
    cols[0].metric("Total Requests", summary["total_requests"])
    cols[1].metric("Inflight", summary["inflight_requests"])
    cols[2].metric("Req/min", summary["recent_rate_per_min"])
    cols[3].metric(
        "P50 ms",
        summary["p50_latency_ms"] if summary["p50_latency_ms"] is not None else "n/a",
    )
    cols[4].metric(
        "P95 ms",
        summary["p95_latency_ms"] if summary["p95_latency_ms"] is not None else "n/a",
    )
    cols[5].metric("Prompt Tokens", summary["prompt_tokens"])
    cols[6].metric("Total Tokens", summary["total_tokens"])

    cols = st.columns(4)
    cols[0].metric("Provider", str(current_run.get("llm_provider") or "-"))
    cols[1].metric("RPM Limit", str(current_run.get("llm_rpm_limit") or "-"))
    cols[2].metric("Models", ", ".join(summary["models"]) or "-")
    cols[3].metric("Errors", summary["error_count"])

    recent = summary["recent_finished"]
    if recent:
        frame = pd.DataFrame(recent).reindex(
            columns=[
                "request_id",
                "latency_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "finish_reason",
                "error_type",
            ]
        )
        st.dataframe(
            frame.fillna(""),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No LLM request logs for this run yet.")

    if summary["recent_errors"]:
        with st.expander("Recent LLM Errors", expanded=False):
            st.json(summary["recent_errors"])


def _render_timeline(snapshot: dict) -> None:
    if snapshot["timeline"]:
        st.dataframe(
            pd.DataFrame(snapshot["timeline"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No run_events.jsonl entries yet. New runs will populate the timeline.")


def _render_candidates(snapshot: dict) -> None:
    rows = snapshot["candidate_rows"]
    if not rows:
        st.info("No candidate events yet.")
        return

    for row in rows[:8]:
        title = row["label"]
        score = row.get("after_score")
        status = "accepted" if row.get("accepted") else "pending"
        with st.expander(
            f"{title} | {status} | score={score if score is not None else 'n/a'}",
            expanded=False,
        ):
            meta_cols = st.columns(4)
            meta_cols[0].metric("Checkpoint OK", row.get("checkpoint_ok"))
            meta_cols[1].metric(
                "Before",
                row.get("before_score")
                if row.get("before_score") is not None
                else "n/a",
            )
            meta_cols[2].metric(
                "After",
                row.get("after_score") if row.get("after_score") is not None else "n/a",
            )
            meta_cols[3].metric(
                "Accepted Score",
                row.get("accepted_score")
                if row.get("accepted_score") is not None
                else "n/a",
            )

            if row.get("reject_reason"):
                st.warning(row["reject_reason"])
            if row.get("error"):
                st.error(row["error"])
            if row.get("rationale"):
                st.caption(row["rationale"])

            detail_path = resolve_detail_file(
                snapshot["run_dir"], snapshot["current_run"], row.get("detail_file")
            )
            detail_payload = load_detail_payload(detail_path)
            if isinstance(detail_payload, dict) and isinstance(
                detail_payload.get("ops"), list
            ):
                preview = patch_preview_from_ops(detail_payload["ops"])
            else:
                preview = patch_preview_from_ops(row.get("ops", []))
            if preview:
                st.code(preview, language="diff")


def main() -> None:
    st.set_page_config(page_title="NOA Runtime Dashboard", layout="wide")
    st.title("NOA Runtime Dashboard")

    refresh_enabled = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_ms = st.sidebar.slider(
        "Refresh (ms)", min_value=500, max_value=5000, value=1000, step=500
    )
    stale_after_sec = st.sidebar.slider(
        "Heartbeat Stale (s)", min_value=5, max_value=60, value=HEARTBEAT_STALE_SEC
    )
    active_window_sec = st.sidebar.slider(
        "Active Run Window (s)",
        min_value=10,
        max_value=300,
        value=ACTIVE_RUN_WINDOW_SEC,
        step=10,
    )
    follow_latest = st.sidebar.checkbox("Follow Latest Active Run", value=True)

    if refresh_enabled:
        st_autorefresh(interval=refresh_ms, key="dashboard_refresh")

    runs = list_runs(ROOT)
    if not runs:
        st.warning(f"No runs found under {ROOT}")
        return

    default_run_id = pick_default_run(runs, active_window_sec=active_window_sec)
    run_ids = [run.run_id for run in runs]

    if follow_latest:
        selected_run_id = default_run_id
        st.sidebar.caption(f"Selected automatically: {selected_run_id}")
    else:
        default_index = (
            run_ids.index(default_run_id) if default_run_id in run_ids else 0
        )
        selected_run_id = st.sidebar.selectbox("Run", run_ids, index=default_index)

    selected = next(run for run in runs if run.run_id == selected_run_id)
    snapshot = build_run_snapshot(
        selected.run_dir,
        stale_after_sec=stale_after_sec,
        active_window_sec=active_window_sec,
        pid_exists=psutil.pid_exists,
    )

    st.caption(str(selected.run_dir))
    _render_overview(snapshot)

    st.subheader("LLM")
    _render_llm(snapshot)

    st.subheader("Timeline")
    _render_timeline(snapshot)

    st.subheader("Candidates")
    _render_candidates(snapshot)


if __name__ == "__main__":
    main()
