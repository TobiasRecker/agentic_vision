from __future__ import annotations

import argparse
from pathlib import Path

from .hybrid_annotation import HybridAnnotationGui
from .hybrid_io import prepare_hybrid_manifest, read_yaml
from .hybrid_pipeline import run_hybrid_reconstruction


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, annotate, and reconstruct a hybrid clip session.")
    parser.add_argument("command", choices=("prepare", "annotate", "reconstruct", "all"))
    parser.add_argument("--session", required=True, help="Clip capture session directory")
    parser.add_argument("--config", default="", help="Optional hybrid reconstruction YAML")
    parser.add_argument("--roi-size", type=int, default=1200, help="Generated lossless ROI size in fullres pixels")
    parser.add_argument("--overwrite-roi", action="store_true")
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)
    session = Path(parsed.session).expanduser().resolve()
    if parsed.command in ("prepare", "all"):
        manifest = prepare_hybrid_manifest(session, parsed.roi_size, parsed.overwrite_roi)
        print(f"Prepared {manifest}")
    if parsed.command in ("annotate", "all"):
        if not (session / "hybrid" / "manifest.json").is_file():
            prepare_hybrid_manifest(session, parsed.roi_size, parsed.overwrite_roi)
        HybridAnnotationGui(session).run()
    if parsed.command in ("reconstruct", "all"):
        config = read_yaml(Path(parsed.config).expanduser().resolve()) if parsed.config else {}
        report = run_hybrid_reconstruction(session, config)
        print(f"Hybrid result accepted={report.get('accepted', False)}")
        for failure in report.get("failures", []):
            print(f"  - {failure}")
        return 0 if report.get("accepted", False) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
