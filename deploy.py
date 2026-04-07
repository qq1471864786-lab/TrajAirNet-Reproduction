from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

SERVER_IP = "10.23.66.99"
USERNAME = "wangzhilin"
PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PATH = f"/home/{USERNAME}/{PROJECT_ROOT.name}"
INCLUDE_EXTS = {'.py', '.yaml', '.yml', '.sh', '.md', '.txt', '.json'}
INCLUDE_FILES = {'Dockerfile', '.gitignore'}
DATA_INCLUDE_EXTS = {'.txt'}
DATA_INCLUDE_ROOTS = (
    Path('dataset/social'),
    Path('dataset/no_social'),
)
EXCLUDE_DIRS = {
    '.git',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.idea',
    '.vscode',
    '.claude',
    'dataset',
    'outputs',
    'save_model',
    'saved_models',
    'results',
    'tmp',
    'tool-outputs',
}
SYNC_MARKER = '.last_sync'
MANIFEST_FILE = '.deploy_manifest.json'
EXCLUDE_FILES = {SYNC_MARKER, MANIFEST_FILE}
SSH_OPTIONS = []


def _default_ssh_key_candidates() -> list[Path]:
    home = Path.home()
    candidates = [
        home / '.ssh' / 'id_rsa',
        Path('C:/Users/14718/.ssh/id_rsa'),
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def resolve_ssh_options() -> list[str]:
    env_key = os.environ.get('TRAJAIR_SSH_KEY', '').strip()
    if env_key:
        key_path = Path(env_key)
        if key_path.exists():
            return ['-o', 'BatchMode=yes', '-i', str(key_path)]

    for key_path in _default_ssh_key_candidates():
        if key_path.exists():
            return ['-o', 'BatchMode=yes', '-i', str(key_path)]
    return list(SSH_OPTIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Deploy changed project files to the remote server.')
    parser.add_argument('--dry-run', action='store_true', help='Only print which files would be uploaded/deleted.')
    parser.add_argument('--force', action='store_true', help='Ignore the saved manifest and upload all tracked files.')
    parser.add_argument('--quiet', action='store_true', help='Reduce per-file output.')
    return parser.parse_args()


def run_command(args: list[str], *, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        capture_output=capture_output,
        text=True,
        encoding='utf-8',
        errors='replace',
    )


def load_last_sync(marker_path: Path) -> float:
    if not marker_path.exists():
        return 0.0
    try:
        return float(marker_path.read_text(encoding='utf-8').strip())
    except ValueError:
        return 0.0


def compute_file_digest(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}
    files = payload.get('files', {})
    if isinstance(files, list):
        return {str(item).replace('\\', '/'): '' for item in files if isinstance(item, str)}
    if not isinstance(files, dict):
        return {}
    manifest: dict[str, str] = {}
    for rel_path, digest in files.items():
        if not isinstance(rel_path, str) or not isinstance(digest, str):
            continue
        manifest[rel_path.replace('\\', '/')] = digest
    return manifest


def save_manifest(manifest_path: Path, fingerprints: dict[str, str]) -> None:
    payload = {
        'updated_at': time.time(),
        'files': {rel_path: fingerprints[rel_path] for rel_path in sorted(fingerprints)},
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding='utf-8')


def is_included(file_path: Path) -> bool:
    if file_path.name in EXCLUDE_FILES:
        return False
    return file_path.name in INCLUDE_FILES or file_path.suffix.lower() in INCLUDE_EXTS


def collect_project_files(local_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for root, dirs, names in os.walk(local_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        root_path = Path(root)
        for name in names:
            path = root_path / name
            if not is_included(path):
                continue
            rel_path = path.relative_to(local_root).as_posix()
            files[rel_path] = path

    for include_root in DATA_INCLUDE_ROOTS:
        abs_root = (local_root / include_root).resolve()
        if not abs_root.exists():
            continue
        for root, _, names in os.walk(abs_root):
            root_path = Path(root)
            for name in names:
                path = root_path / name
                if path.suffix.lower() not in DATA_INCLUDE_EXTS:
                    continue
                rel_path = path.relative_to(local_root).as_posix()
                files[rel_path] = path
    return files


def detect_changed_files(
    current_files: dict[str, Path],
    previous_fingerprints: dict[str, str],
    last_sync_time: float,
) -> tuple[list[str], dict[str, str]]:
    changed_files: list[str] = []
    current_fingerprints: dict[str, str] = {}
    manifest_missing = not previous_fingerprints

    for rel_path in sorted(current_files):
        local_file = current_files[rel_path]
        digest = compute_file_digest(local_file)
        current_fingerprints[rel_path] = digest
        previous_digest = previous_fingerprints.get(rel_path)
        if previous_digest == digest:
            continue
        mtime = local_file.stat().st_mtime
        if previous_digest == '' and mtime <= last_sync_time:
            continue
        if manifest_missing and previous_digest is None and last_sync_time > 0.0 and mtime <= last_sync_time:
            continue
        changed_files.append(rel_path)

    return changed_files, current_fingerprints


def ensure_remote_dir(remote_dir: str) -> None:
    ssh_options = resolve_ssh_options()
    run_command(['ssh', *ssh_options, f'{USERNAME}@{SERVER_IP}', f'mkdir -p {shlex.quote(remote_dir)}'])


def upload_file(local_file: Path, remote_file: str) -> None:
    ssh_options = resolve_ssh_options()
    run_command(['scp', '-q', *ssh_options, str(local_file), f'{USERNAME}@{SERVER_IP}:{remote_file}'])


def delete_remote_file(remote_file: str) -> None:
    ssh_options = resolve_ssh_options()
    run_command(['ssh', *ssh_options, f'{USERNAME}@{SERVER_IP}', f'rm -f {shlex.quote(remote_file)}'])


def print_preview(changed_files: list[str], deleted_files: list[str]) -> None:
    if changed_files:
        print('[deploy] upload queue:')
        for rel_path in changed_files:
            print(f'  + {rel_path}')
    if deleted_files:
        print('[deploy] delete queue:')
        for rel_path in deleted_files:
            print(f'  - {rel_path}')


def print_sync_progress(*, action: str, index: int, total: int, rel_path: str, quiet: bool) -> None:
    if quiet:
        return
    icon = '🚀' if action == 'upload' else '🧹'
    print(f'[{icon} {action} {index}/{total}] {rel_path}')


def deploy(args: argparse.Namespace) -> int:
    local_root = PROJECT_ROOT
    marker_path = local_root / SYNC_MARKER
    manifest_path = local_root / MANIFEST_FILE
    last_sync_time = 0.0 if args.force else load_last_sync(marker_path)
    start_time = time.time()

    current_files = collect_project_files(local_root)
    previous_fingerprints = {} if args.force else load_manifest(manifest_path)
    changed_files, current_fingerprints = detect_changed_files(current_files, previous_fingerprints, last_sync_time)
    deleted_files = [] if args.force else sorted(set(previous_fingerprints) - set(current_files))

    print(f'[deploy] target={USERNAME}@{SERVER_IP}:{REMOTE_PATH}')
    print(f'[deploy] tracked={len(current_files)} changed={len(changed_files)} deleted={len(deleted_files)} force={args.force}')

    if not changed_files and not deleted_files:
        print('[deploy] no changes detected; nothing to sync.')
        return 0

    print_preview(changed_files, deleted_files)
    if args.dry_run:
        print('[deploy] dry-run only; no files were transferred.')
        return 0

    failures: list[str] = []
    uploaded_count = 0
    deleted_count = 0
    total_uploads = len(changed_files)
    total_deletes = len(deleted_files)

    for index, rel_path in enumerate(changed_files, start=1):
        local_file = current_files[rel_path]
        remote_file = f"{REMOTE_PATH.rstrip('/')}/{rel_path}"
        remote_dir = remote_file.rsplit('/', 1)[0]
        try:
            print_sync_progress(action='upload', index=index, total=total_uploads, rel_path=rel_path, quiet=args.quiet)
            ensure_remote_dir(remote_dir)
            upload_file(local_file, remote_file)
            uploaded_count += 1
        except subprocess.CalledProcessError as exc:
            failures.append(f'upload {rel_path}: {exc}')
            print(f'[error] upload failed: {rel_path}', file=sys.stderr)

    for index, rel_path in enumerate(deleted_files, start=1):
        remote_file = f"{REMOTE_PATH.rstrip('/')}/{rel_path}"
        try:
            print_sync_progress(action='delete', index=index, total=total_deletes, rel_path=rel_path, quiet=args.quiet)
            delete_remote_file(remote_file)
            deleted_count += 1
        except subprocess.CalledProcessError as exc:
            failures.append(f'delete {rel_path}: {exc}')
            print(f'[error] delete failed: {rel_path}', file=sys.stderr)

    if failures:
        print(f'[deploy] finished with {len(failures)} failure(s); sync marker was not advanced.', file=sys.stderr)
        for item in failures:
            print(f'  - {item}', file=sys.stderr)
        return 1

    marker_path.write_text(str(start_time), encoding='utf-8')
    save_manifest(manifest_path, current_fingerprints)
    elapsed = time.time() - start_time
    print(f'[deploy] uploaded={uploaded_count} deleted={deleted_count} failures=0')
    print(f'[deploy] ✅ sync completed successfully in {elapsed:.2f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(deploy(parse_args()))
