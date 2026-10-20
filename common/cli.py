"""Developer command line: ``aip <command>``.

Phase 0 commands:
  aip doctor            hardware check + recommended run mode
  aip config show       print the fully resolved config (YAML + env vars)
  aip config validate   validate one or more YAML files
  aip paths             show and create the data directories

Later phases add ``aip track``, ``aip voice``, ``aip bench`` ... to this same entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from common import __version__
from common.config import Settings, default_config_files, load_settings
from common.system_info import system_report


def _load(args: argparse.Namespace) -> Settings:
    files = [Path(p) for p in args.config] if args.config else default_config_files()
    return load_settings(files, missing_ok=not args.config)


def cmd_doctor(_: argparse.Namespace) -> int:
    rep = system_report()
    print(f"AI Presenter {__version__}")
    print(f"Python  : {rep['python']}")
    print(f"OS      : {rep['os']}")
    print(f"CPU     : {rep['cpu']}")
    if not rep["gpus"]:
        print("GPU     : none detected")
    for g in rep["gpus"]:
        cc = ".".join(map(str, g["compute_cap"])) if g["compute_cap"] else "?"
        print(
            f"GPU {g['index']}   : {g['name']}  {g['vram_total_mb']} MB  "
            f"driver {g['driver']}  compute {cc}"
        )
    print(f"Mode    : {rep['recommended_mode'].upper()} - {rep['reason']}")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    data = json.loads(_load(args).model_dump_json())
    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), end="")
    return 0


def cmd_config_validate(args: argparse.Namespace) -> int:
    try:
        load_settings([Path(p) for p in args.files], missing_ok=False)
    except (ValidationError, ValueError, FileNotFoundError) as e:
        print(f"INVALID:\n{e}", file=sys.stderr)
        return 1
    print("OK")
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    paths = _load(args).paths.ensure()
    print(f"root        {paths.root}")
    for d in paths.all_dirs():
        print(f"{d.name:<11} {d}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aip", description="AI Presenter developer tools")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument(
        "-c", "--config", action="append", metavar="FILE", help="YAML config file (repeatable)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check hardware").set_defaults(func=cmd_doctor)
    sub.add_parser("paths", help="show/create data dirs").set_defaults(func=cmd_paths)

    cfg = sub.add_parser("config", help="config tools")
    cfg_sub = cfg.add_subparsers(dest="config_cmd", required=True)
    cfg_sub.add_parser("show", help="print resolved config").set_defaults(func=cmd_config_show)
    val = cfg_sub.add_parser("validate", help="validate YAML files")
    val.add_argument("files", nargs="+")
    val.set_defaults(func=cmd_config_validate)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValidationError as e:
        print(f"Config error:\n{e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
