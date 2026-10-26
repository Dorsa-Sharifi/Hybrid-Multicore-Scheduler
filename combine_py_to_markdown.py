#!/usr/bin/env python3
"""Combine all Python files in a project into a single Markdown document."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect every .py file under a directory and write them into one Markdown file."
    )
    parser.add_argument(
        "-r",
        "--root",
        default=".",
        help="Project root to scan (default: current directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="all_python_code.md",
        help="Output Markdown file path (default: all_python_code.md).",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include files inside hidden directories (like .git or .venv).",
    )
    return parser.parse_args()


def is_hidden_path(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def collect_python_files(root: Path, include_hidden: bool) -> list[Path]:
    files: list[Path] = []
    for file_path in root.rglob("*.py"):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(root)
        if not include_hidden and is_hidden_path(rel):
            continue
        files.append(file_path)
    return sorted(files, key=lambda p: str(p.relative_to(root)))


def build_markdown(root: Path, python_files: list[Path]) -> str:
    lines: list[str] = ["# Combined Python Source", ""]

    for file_path in python_files:
        rel = file_path.relative_to(root)
        content = file_path.read_text(encoding="utf-8", errors="replace")
        lines.extend(
            [
                f"## `{rel}`",
                "",
                "```python",
                content.rstrip("\n"),
                "```",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()

    python_files = collect_python_files(root=root, include_hidden=args.include_hidden)

    # Do not include this script itself if it is inside the scanned root.
    this_script = Path(__file__).resolve()
    python_files = [p for p in python_files if p.resolve() != this_script]

    markdown = build_markdown(root=root, python_files=python_files)
    output.write_text(markdown, encoding="utf-8")

    print(f"Wrote {len(python_files)} Python files to {output}")


if __name__ == "__main__":
    main()
