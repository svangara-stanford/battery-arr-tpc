#!/usr/bin/env python3
"""Monitor the sealed Track A Sherlock campaign — single-pass, idempotent.

Designed to be re-invoked on a schedule (e.g. by the harness `/loop` skill
every 10 minutes). Each invocation:

  1. Reads the campaign state file (`$SCRATCH/battery-arr/launch_logs/trackA_state.json`).
  2. Polls `squeue` for pending/running jobs and `sacct` for terminal state.
  3. Classifies each terminal job: SUCCESS, OOM, TIMEOUT, NODE_FAIL,
     PREEMPTED, LLM_TRANSIENT, DATA_FIXUP, CODE_BUG, UNKNOWN.
  4. Auto-resubmits resource-class / transient / data-class failures via
     ``sbatch``, bumping memory or wall-time as appropriate. Updates state.
  5. Flags CODE_BUG / UNKNOWN with ``needs_human=True`` and stops touching
     them — those require human review.
  6. Prints a concise campaign-progress block to stdout.

The script never mutates the repo source tree. It edits only the state JSON
and submits new Slurm jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_STATE = "/scratch/users/svangara/battery-arr/launch_logs/trackA_state.json"
REPO_ROOT = Path("/home/groups/darve/svangara/battery-arr-tpc")
SLURM_LOG_DIR = REPO_ROOT / "runs" / "slurm"

# Failure-class buckets the loop is allowed to handle automatically.
RESOURCE_CLASS = {"OOM", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
TRANSIENT_CLASS = {"LLM_TRANSIENT", "DATA_FIXUP"}
AUTO_FIX_CLASSES = RESOURCE_CLASS | TRANSIENT_CLASS

# Patterns suggesting a transient LLM API error in the .out file.
LLM_TRANSIENT_PATTERNS = [
    re.compile(r"openai\.(?:RateLimit|APITimeout|APIConnection|InternalServer)Error", re.I),
    re.compile(r"\b(?:429|500|502|503|504)\b.*?(?:openai|aiapi-prod\.stanford)", re.I),
    re.compile(r"Connection (?:reset by peer|aborted|refused)", re.I),
    re.compile(r"Read timed out", re.I),
    re.compile(r"openai\.APIStatusError.*\b5\d\d\b", re.I),
]

# Patterns suggesting a fixable data/scratch issue (re-running on a fresh node usually heals).
DATA_FIXUP_PATTERNS = [
    re.compile(r"No space left on device", re.I),
    re.compile(r"\.mat: cannot copy", re.I),
    re.compile(r"Stale file handle", re.I),
    re.compile(r"Input/output error", re.I),
    re.compile(r"\[Errno 116\]", re.I),  # Stale file handle on NFS
]

# Patterns that are almost always code-level — flag for human.
CODE_BUG_PATTERNS = [
    re.compile(r"pydantic_core\._pydantic_core\.ValidationError", re.I),
    re.compile(r"^ValueError: Unsupported (?:feature_set|target_transform)", re.M),
    re.compile(r"^TypeError:", re.M),
    re.compile(r"^KeyError:", re.M),
    re.compile(r"^AttributeError:", re.M),
    re.compile(r"^AssertionError", re.M),
    re.compile(r"^ImportError:", re.M),
    re.compile(r"^ModuleNotFoundError:", re.M),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def _sacct(job_ids: list[str]) -> dict[str, dict[str, str]]:
    """Return {job_id: {State, ExitCode, Elapsed, MaxRSS}} for terminal queries."""
    if not job_ids:
        return {}
    cmd = [
        "sacct", "-j", ",".join(job_ids),
        "--format=JobID,State,ExitCode,Elapsed,MaxRSS",
        "-P", "--noheader",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=60)
    except subprocess.SubprocessError as exc:
        print(f"sacct failed: {exc}", file=sys.stderr)
        return {}
    rows: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        jobid, state = parts[0], parts[1]
        # We only care about the parent job rows (no .batch/.extern suffix).
        if "." in jobid:
            continue
        rows[jobid] = {
            "State": state.strip(),
            "ExitCode": parts[2].strip() if len(parts) > 2 else "",
            "Elapsed": parts[3].strip() if len(parts) > 3 else "",
            "MaxRSS": parts[4].strip() if len(parts) > 4 else "",
        }
    return rows


def _squeue(job_ids: list[str]) -> dict[str, dict[str, str]]:
    """Return {job_id: {State, Reason, Elapsed}} for jobs still in queue."""
    if not job_ids:
        return {}
    cmd = ["squeue", "-j", ",".join(job_ids), "-h", "-o", "%i|%T|%R|%M|%L"]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=60)
    except subprocess.SubprocessError:
        return {}
    rows: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        rows[parts[0]] = {
            "State": parts[1],
            "Reason": parts[2],
            "Elapsed": parts[3],
            "TimeLeft": parts[4],
        }
    return rows


def _classify_failure(state: str, log_text: str) -> str:
    state = state.upper()
    if state.startswith("OUT_OF_MEMORY"):
        return "OOM"
    if state.startswith("TIMEOUT"):
        return "TIMEOUT"
    if state.startswith("NODE_FAIL"):
        return "NODE_FAIL"
    if state.startswith("BOOT_FAIL"):
        return "BOOT_FAIL"
    if state.startswith("PREEMPTED"):
        return "PREEMPTED"
    if state.startswith("DEADLINE"):
        return "DEADLINE"
    # State == FAILED, CANCELLED, or COMPLETED with non-zero exit: inspect log.
    for pat in CODE_BUG_PATTERNS:
        if pat.search(log_text):
            return "CODE_BUG"
    for pat in LLM_TRANSIENT_PATTERNS:
        if pat.search(log_text):
            return "LLM_TRANSIENT"
    for pat in DATA_FIXUP_PATTERNS:
        if pat.search(log_text):
            return "DATA_FIXUP"
    return "UNKNOWN"


def _job_log_path(job_id: str) -> Path:
    return SLURM_LOG_DIR / f"oba-trackA-{job_id}.out"


def _tail_log(job_id: str, max_chars: int = 200_000) -> str:
    log = _job_log_path(job_id)
    if not log.exists():
        return ""
    text = log.read_text(errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text


def _result_json_exists(job_id: str, recipe: str, split_mode: str, seed: int, reports_root: Path) -> bool:
    return (reports_root / f"trackA_{recipe}_{split_mode}_seed{seed}_{job_id}"
            / "role_agent_workflow.json").exists()


def _bump_mem(mem_gb: int) -> int:
    bumped = mem_gb * 2
    return min(bumped, 256)


def _bump_time(time_str: str) -> str:
    # HH:MM:SS -> double hours, cap at 7 days
    parts = time_str.split(":")
    if len(parts) != 3:
        return "12:00:00"
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    total = h * 3600 + m * 60 + s
    doubled = min(total * 2, 7 * 24 * 3600 - 60)
    h2, rem = divmod(doubled, 3600)
    m2, s2 = divmod(rem, 60)
    return f"{h2:d}:{m2:02d}:{s2:02d}"


def _resubmit(job: dict[str, Any], reason: str, dry_run: bool = False) -> tuple[bool, str]:
    cmd = [
        "sbatch",
        f"--cpus-per-task={job['cpus']}",
        f"--mem={job['mem_gb']}G",
        f"--time={job['time']}",
        "--export=ALL,"
        f"OFFLINE=false,"
        f"FEATURE_RECIPE={job['recipe']},"
        f"SPLIT_MODE={job['split_mode']},"
        f"SPLIT_SEED={job['split_seed']},"
        f"ITERATIONS={job['iterations']},"
        f"CANDIDATES_PER_ITERATION={job['candidates_per_iteration']},"
        f"DO_FINAL_BATCH9=false,"
        f"FINAL_BATCH9_TOP_K=0",
        "nersc/sherlock_trackA_discovery.slurm",
    ]
    if dry_run:
        return False, f"[dry-run] {shlex.join(cmd)}  (reason={reason})"
    try:
        out = subprocess.check_output(cmd, text=True, cwd=str(REPO_ROOT), timeout=60)
    except subprocess.SubprocessError as exc:
        return False, f"sbatch failed: {exc}"
    parts = out.strip().split()
    if not parts:
        return False, f"sbatch returned no job id: {out!r}"
    return True, parts[-1]


def _handle_terminal_job(job: dict[str, Any], sacct_state: str, dry_run: bool) -> dict[str, Any]:
    """Mutate ``job`` in place; return a per-job action log entry."""
    current = job["current_job_id"]
    log_text = _tail_log(current)
    cls = _classify_failure(sacct_state, log_text)
    entry = {
        "at": _utc_now(),
        "job_id": current,
        "state": sacct_state,
        "class": cls,
    }
    job["last_state"] = sacct_state
    job["history"].append(entry)

    if sacct_state.upper().startswith("COMPLETED"):
        entry["action"] = "completed"
        return entry

    if job["resubmit_count"] >= job["max_resubmits"]:
        job["needs_human"] = True
        job["human_reason"] = f"exceeded max_resubmits={job['max_resubmits']} (last class={cls})"
        entry["action"] = "halt-max-resubmits"
        return entry

    if cls not in AUTO_FIX_CLASSES:
        job["needs_human"] = True
        job["human_reason"] = f"non-auto-fixable failure class {cls}; last state {sacct_state}"
        entry["action"] = "halt-needs-human"
        return entry

    # Bump resources for OOM / TIMEOUT; leave alone for node/transient.
    if cls == "OOM":
        old = job["mem_gb"]
        job["mem_gb"] = _bump_mem(old)
        entry["mem_bump"] = f"{old}G -> {job['mem_gb']}G"
    elif cls in {"TIMEOUT", "DEADLINE"}:
        old = job["time"]
        job["time"] = _bump_time(old)
        entry["time_bump"] = f"{old} -> {job['time']}"

    ok, info = _resubmit(job, reason=cls, dry_run=dry_run)
    if ok:
        job["current_job_id"] = info
        job["attempts"].append(info)
        job["resubmit_count"] += 1
        job["last_state"] = "RESUBMITTED"
        entry["action"] = "resubmitted"
        entry["new_job_id"] = info
    else:
        job["needs_human"] = True
        job["human_reason"] = info
        entry["action"] = "resubmit-failed"
        entry["info"] = info
    return entry


def _summarize(state: dict[str, Any]) -> None:
    jobs = state["jobs"]
    by_state: dict[str, int] = {}
    completed: list[dict[str, Any]] = []
    needs_human: list[dict[str, Any]] = []
    for job in jobs:
        by_state[job["last_state"]] = by_state.get(job["last_state"], 0) + 1
        if job["last_state"].upper().startswith("COMPLETED"):
            completed.append(job)
        if job["needs_human"]:
            needs_human.append(job)
    print(f"=== Track A campaign monitor @ {_utc_now()} ===")
    print(f"campaign         : {state.get('campaign')}")
    print(f"state file       : {state.get('_state_path', '?')}")
    print(f"total jobs       : {len(jobs)}")
    print(f"completed (OK)   : {len(completed)}")
    print(f"needs human      : {len(needs_human)}")
    print("state breakdown  :", json.dumps(by_state, indent=None))
    print()
    if needs_human:
        print("!! NEEDS HUMAN REVIEW:")
        for job in needs_human:
            print(f"  - {job['recipe']:14s} {job['split_mode']:6s} seed={job['split_seed']} "
                  f"current_job={job['current_job_id']} attempts={len(job['attempts'])} "
                  f"reason={job['human_reason']}")
        print()
    # Per-job one-liner
    for job in jobs:
        marker = "X" if job["needs_human"] else ("." if job["last_state"].upper().startswith("COMPLETED") else "-")
        print(f" {marker} {job['recipe']:14s} {job['split_mode']:6s} seed={job['split_seed']} "
              f"job={job['current_job_id']:>10s} state={job['last_state']:<14s} "
              f"resubs={job['resubmit_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=DEFAULT_STATE, help="Path to trackA_state.json")
    parser.add_argument("--dry-run", action="store_true", help="Do not call sbatch; just print actions.")
    args = parser.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        sys.exit(f"ERROR: missing state file {state_path}")
    state = _read_state(state_path)
    state["_state_path"] = str(state_path)
    state["_last_poll"] = _utc_now()

    active_ids = [j["current_job_id"] for j in state["jobs"]
                  if not j["needs_human"]
                  and not j["last_state"].upper().startswith("COMPLETED")]
    sq = _squeue(active_ids) if active_ids else {}
    sa = _sacct(active_ids) if active_ids else {}

    for job in state["jobs"]:
        if job["needs_human"] or job["last_state"].upper().startswith("COMPLETED"):
            continue
        jid = job["current_job_id"]
        if jid in sq:
            job["last_state"] = sq[jid]["State"]
            continue
        # Not in queue any more -> consult sacct
        row = sa.get(jid)
        if not row:
            # sacct hasn't observed it yet; leave as-is.
            continue
        terminal_state = row["State"]
        if terminal_state in {"RUNNING", "PENDING", "REQUEUED", "RESIZING", "SUSPENDED"}:
            job["last_state"] = terminal_state
            continue
        _handle_terminal_job(job, terminal_state, dry_run=args.dry_run)

    _write_state(state_path, state)
    _summarize(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
