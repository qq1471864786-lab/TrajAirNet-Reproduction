import argparse
import json
import os
import time
from pathlib import Path

from remote_common import add_remote_project_root_arg, add_remote_target_args, upload_file


INCLUDE_EXTS = {".py", ".md", ".txt", ".yaml", ".yml", ".sh", ".toml"}
INCLUDE_NAMES = {".gitignore", "README.md", "requirements.txt"}
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "dataset",
    "save_model",
    "tmp",
    "tmp_smoke_align",
    "tmp_smoke_protobasis",
    ".remote_runs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync ProtoBasis-Net source files to the remote server.")
    add_remote_target_args(parser)
    add_remote_project_root_arg(parser, flag="--remote-root")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--full", action="store_true", help="Sync all included files instead of only modified files.")
    return parser.parse_args()


def marker_path(project_root: Path) -> Path:
    return project_root / ".remote_sync_state.json"


def load_last_sync(project_root: Path) -> float:
    path = marker_path(project_root)
    if not path.exists():
        return 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload.get("last_sync", 0.0))
    except Exception:
        return 0.0


def save_last_sync(project_root: Path, timestamp: float, synced_files: list[str]) -> None:
    payload = {
        "last_sync": timestamp,
        "synced_files": synced_files,
    }
    marker_path(project_root).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def should_include(path: Path) -> bool:
    if path.name in INCLUDE_NAMES:
        return True
    return path.suffix.lower() in INCLUDE_EXTS


def collect_files(project_root: Path, full: bool) -> list[Path]:
    cutoff = load_last_sync(project_root)
    selected: list[Path] = []
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if not should_include(path):
                continue
            if not full and path.stat().st_mtime <= cutoff:
                continue
            selected.append(path)
    return sorted(selected)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    files = collect_files(project_root, args.full)
    if not files:
        print("没有检测到需要同步的文件。")
        return 0

    synced_rel: list[str] = []
    for local_file in files:
        rel = local_file.relative_to(project_root).as_posix()
        remote_file = f"{args.remote_root}/{rel}"
        upload_file(
            local_file,
            remote_file,
            user=args.user,
            host=args.host,
            port=args.port,
            password_env=args.password_env,
        )
        synced_rel.append(rel)

    save_last_sync(project_root, time.time(), synced_rel)
    print(f"同步完成，共 {len(synced_rel)} 个文件。")
    for rel in synced_rel:
        print(rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
