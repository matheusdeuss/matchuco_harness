"""Snapshots of a file taken immediately before `write` or `edit` changes it.

The bytes live here, one JSON file per snapshot, under
`.matchuco/checkpoints/<id>.json`. The session log only stores the id (see
`sessions/`). Keeping the payload out of the JSONL means a rewind can still
see every version without making every transcript line the size of the file.

A snapshot records whether the file existed. Restoring a file that did not
exist deletes it; restoring one that did writes the old bytes back and tries
to put the previous mtime back, so the read-before-write check still
recognises the file the model already read.
"""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path

from pydantic import BaseModel


class CheckpointError(Exception):
    """A snapshot cannot be saved or restored."""


class CheckpointRecord(BaseModel):
    id: str
    path: str  # workspace-relative, POSIX
    existed: bool
    mtime: float | None = None
    content_b64: str = ""


class CheckpointStore:
    """The `.matchuco/checkpoints/` directory of one workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / ".matchuco" / "checkpoints"

    def capture(self, path: Path) -> CheckpointRecord:
        """Snapshot `path` as it is right now. Missing files are recorded as such."""
        relative = self._relative(path)
        target = self._inside(relative)
        existed = target.is_file()
        if target.exists() and not existed:
            raise CheckpointError(f"{relative} is not a regular file")
        mtime = target.stat().st_mtime if existed else None
        payload = target.read_bytes() if existed else b""
        record = CheckpointRecord(
            id=uuid.uuid4().hex[:12],
            path=relative,
            existed=existed,
            mtime=mtime,
            content_b64=base64.b64encode(payload).decode("ascii"),
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{record.id}.json"
        destination.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        return record

    def load(self, checkpoint_id: str) -> CheckpointRecord:
        path = self.directory / f"{checkpoint_id}.json"
        if not path.is_file():
            raise CheckpointError(f"no checkpoint {checkpoint_id!r}")
        try:
            return CheckpointRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError as e:
            raise CheckpointError(f"{path}: {e}") from e

    def restore(self, checkpoint_id: str) -> None:
        """Put the file back to the snapshotted bytes, or delete it if it was new."""
        record = self.load(checkpoint_id)
        target = self._inside(record.path)
        if record.existed:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(record.content_b64))
            if record.mtime is not None:
                os.utime(target, (record.mtime, record.mtime))
            return
        if target.is_file():
            target.unlink()
        elif target.exists():
            raise CheckpointError(f"{record.path} exists and is not a file; not deleting it")

    def _relative(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.root)
        except ValueError as e:
            raise CheckpointError(f"path {path} is outside the workspace ({self.root})") from e
        text = relative.as_posix()
        if not text or text == ".":
            raise CheckpointError("refusing to checkpoint the workspace root")
        return text

    def _inside(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if target != self.root and self.root not in target.parents:
            raise CheckpointError(f"path {relative!r} escapes the workspace")
        return target
