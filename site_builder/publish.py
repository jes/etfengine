"""Publish site snapshots under public/builds/ and mirror latest to public/."""

from __future__ import annotations

import html
import shutil
from datetime import UTC, datetime
from pathlib import Path


def build_timestamp(moment: datetime | None = None) -> str:
    when = moment or datetime.now(tz=UTC)
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_build_snapshots(builds_dir: Path) -> list[str]:
    if not builds_dir.is_dir():
        return []
    snapshots = [
        path.name
        for path in builds_dir.iterdir()
        if path.is_dir() and (path / "index.html").is_file()
    ]
    return sorted(snapshots, reverse=True)


def publish_snapshot_to_root(snapshot_dir: Path, public_dir: Path) -> None:
    public_dir.mkdir(parents=True, exist_ok=True)
    for path in snapshot_dir.rglob("*"):
        if not path.is_file():
            continue
        target = public_dir / path.relative_to(snapshot_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def write_builds_index(builds_dir: Path) -> None:
    builds_dir.mkdir(parents=True, exist_ok=True)
    snapshots = list_build_snapshots(builds_dir)
    lines = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>ETF Engine — previous builds</title>",
        "</head>",
        "<body>",
        "<h1>Previous builds</h1>",
        "<p><a href=\"/\">Current site</a></p>",
    ]
    if snapshots:
        lines.append("<ul>")
        for snapshot_id in snapshots:
            lines.append(
                "<li>"
                f"<a href=\"{html.escape(snapshot_id)}/index.html\">"
                f"{html.escape(snapshot_id)}</a>"
                "</li>"
            )
        lines.append("</ul>")
    else:
        lines.append("<p>No archived builds yet.</p>")
    lines.extend(["</body>", "</html>"])
    (builds_dir / "index.html").write_text("\n".join(lines) + "\n", encoding="utf-8")
