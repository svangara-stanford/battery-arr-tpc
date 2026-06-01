from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import AgentRole


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _role_value(role: AgentRole | str | None) -> str | None:
    if role is None:
        return None
    if isinstance(role, AgentRole):
        return role.value
    return str(role)


class TraceLogger:
    """Append-only JSONL trace logger for workflow events."""

    def __init__(self, trace_dir: str | Path, run_id: str):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.events_path = self.trace_dir / "events.jsonl"
        self.tool_calls_path = self.trace_dir / "tool_calls.jsonl"
        self.agent_messages_path = self.trace_dir / "agent_messages.jsonl"
        for path in (self.events_path, self.tool_calls_path, self.agent_messages_path):
            path.touch(exist_ok=True)

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("timestamp", _timestamp())
        payload["agent_role"] = _role_value(payload.get("agent_role"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def log_event(
        self,
        *,
        event_type: str,
        iteration: int | None = None,
        agent_role: AgentRole | str | None = None,
        input_artifact_ids: list[str] | None = None,
        output_artifact_ids: list[str] | None = None,
        duration_ms: float | None = None,
        success: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "iteration": iteration,
            "agent_role": agent_role,
            "event_type": event_type,
            "input_artifact_ids": input_artifact_ids or [],
            "output_artifact_ids": output_artifact_ids or [],
            "duration_ms": duration_ms,
            "success": success,
            "error_type": error_type,
            "error_message": error_message,
        }
        if extra:
            record.update(extra)
        self._append(self.events_path, record)

    def log_tool_call(
        self,
        *,
        tool_name: str,
        tool_call_id: str | None = None,
        event_type: str = "tool_call",
        iteration: int | None = None,
        agent_role: AgentRole | str | None = None,
        input_artifact_ids: list[str] | None = None,
        output_artifact_ids: list[str] | None = None,
        duration_ms: float | None = None,
        success: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
        arguments_summary: dict[str, Any] | None = None,
    ) -> None:
        self._append(
            self.tool_calls_path,
            {
                "iteration": iteration,
                "agent_role": agent_role,
                "event_type": event_type,
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "input_artifact_ids": input_artifact_ids or [],
                "output_artifact_ids": output_artifact_ids or [],
                "duration_ms": duration_ms,
                "success": success,
                "error_type": error_type,
                "error_message": error_message,
                "arguments_summary": arguments_summary or {},
            },
        )

    def log_agent_message(
        self,
        *,
        event_type: str,
        iteration: int | None,
        agent_role: AgentRole | str,
        agent_id: str | None = None,
        input_artifact_ids: list[str] | None = None,
        output_artifact_ids: list[str] | None = None,
        duration_ms: float | None = None,
        success: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
        message_summary: str | None = None,
    ) -> None:
        self._append(
            self.agent_messages_path,
            {
                "iteration": iteration,
                "agent_role": agent_role,
                "agent_id": agent_id,
                "event_type": event_type,
                "input_artifact_ids": input_artifact_ids or [],
                "output_artifact_ids": output_artifact_ids or [],
                "duration_ms": duration_ms,
                "success": success,
                "error_type": error_type,
                "error_message": error_message,
                "message_summary": message_summary,
            },
        )
