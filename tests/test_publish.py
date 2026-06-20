from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from site_builder.publish import (
    list_build_snapshots,
    publish_snapshot_to_root,
    write_builds_index,
)


class PublishTests(unittest.TestCase):
    def test_publish_snapshot_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public_dir = root / "public"
            builds_dir = public_dir / "builds"
            snapshot_dir = builds_dir / "2026-06-10T12:00:00Z"
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "index.html").write_text("<html>snap</html>", encoding="utf-8")
            (snapshot_dir / "equity.png").write_bytes(b"png")
            nested = snapshot_dir / "sparklines"
            nested.mkdir()
            (nested / "gold.png").write_bytes(b"png")

            publish_snapshot_to_root(snapshot_dir, public_dir)
            write_builds_index(builds_dir)

            self.assertTrue((public_dir / "index.html").is_file())
            self.assertTrue((public_dir / "sparklines" / "gold.png").is_file())
            self.assertEqual(
                list_build_snapshots(builds_dir),
                ["2026-06-10T12:00:00Z"],
            )
            index = (builds_dir / "index.html").read_text(encoding="utf-8")
            self.assertIn("2026-06-10T12:00:00Z", index)
            self.assertIn('href="/"', index)
