import argparse
import json
import shlex
import sys

from remote_common import add_remote_conda_args, add_remote_project_root_arg, add_remote_target_args, run_target_command


REMOTE_SETUP_SCRIPT = r"""
import json
import os
import shlex
import subprocess
import sys

project_root = os.environ["PROJECT_ROOT"]
conda_sh = os.environ["CONDA_SH"]
conda_env = os.environ["CONDA_ENV"]
python_version = os.environ["PYTHON_VERSION"]
requirements = json.loads(os.environ["REQUIREMENTS_JSON"])
torch_index_url = os.environ.get("TORCH_INDEX_URL", "").strip()
recreate = os.environ.get("RECREATE_ENV") == "1"

python_requirements = [item for item in requirements if item.lower().startswith("python")]
torch_requirements = [item for item in requirements if item.lower().startswith("torch==")]
other_requirements = [
    item for item in requirements
    if item not in python_requirements and item not in torch_requirements
]


def run_conda(command: str) -> subprocess.CompletedProcess:
    shell = f"source {shlex.quote(conda_sh)} && {command}"
    return subprocess.run(["bash", "-lc", shell], capture_output=True, text=True)


def run_checked(command: str, *, step: str) -> str:
    completed = run_conda(command)
    if completed.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "step": step,
                    "command": command,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "returncode": completed.returncode,
                },
                ensure_ascii=False,
            )
        )
    return completed.stdout


env_list = json.loads(run_checked("conda env list --json", step="list-envs"))
env_exists = any(path.endswith(f"/{conda_env}") for path in env_list.get("envs", []))
env_preexisting = env_exists

if recreate and env_exists:
    run_checked(f"conda remove -n {shlex.quote(conda_env)} --all -y", step="remove-env")
    env_exists = False

if not env_exists:
    run_checked(
        f"conda create -n {shlex.quote(conda_env)} python={shlex.quote(python_version)} -y",
        step="create-env",
    )

run_checked(
    f"conda run -n {shlex.quote(conda_env)} python -m pip install --upgrade pip",
    step="upgrade-pip",
)

if other_requirements:
    reqs = " ".join(shlex.quote(item) for item in other_requirements)
    run_checked(
        f"conda run -n {shlex.quote(conda_env)} python -m pip install {reqs}",
        step="install-requirements",
    )

if torch_requirements:
    reqs = " ".join(shlex.quote(item) for item in torch_requirements)
    extra = f" --index-url {shlex.quote(torch_index_url)}" if torch_index_url else ""
    run_checked(
        f"conda run -n {shlex.quote(conda_env)} python -m pip install {reqs}{extra}",
        step="install-torch",
    )

run_checked(
    f"git config --global --add safe.directory {shlex.quote(project_root)}",
    step="git-safe-directory",
)

verify_code = (
    "import json, sys, geographiclib, metar, numpy, pandas, scipy, torch, tqdm, "
    "model.data, model.losses, model.proto_basis_flight_model; "
    "print(json.dumps({"
    "'python': sys.version.split()[0], "
    "'torch': torch.__version__, "
    "'torch_cuda': torch.version.cuda, "
    "'cuda_available': bool(torch.cuda.is_available()), "
    "'cuda_device_count': int(torch.cuda.device_count())"
    "}, ensure_ascii=False))"
)
verify_imports = run_checked(
    f"cd {shlex.quote(project_root)} && conda run -n {shlex.quote(conda_env)} "
    f"python -c {shlex.quote(verify_code)}",
    step="verify-imports",
)
verify_lines = [line.strip() for line in verify_imports.splitlines() if line.strip()]
if not verify_lines:
    raise RuntimeError(
        json.dumps(
            {
                "step": "verify-imports",
                "stdout": verify_imports,
                "stderr": "",
                "returncode": 0,
            },
            ensure_ascii=False,
        )
    )

run_checked(
    f"cd {shlex.quote(project_root)} && conda run -n {shlex.quote(conda_env)} python train.py --help >/dev/null",
    step="verify-train-help",
)

git_head = run_checked(
    f"cd {shlex.quote(project_root)} && git rev-parse --short HEAD",
    step="git-head",
).strip()

payload = {
    "project_root": project_root,
    "conda_env": conda_env,
    "python_version_requested": python_version,
    "python_requirements": python_requirements,
    "other_requirements": other_requirements,
    "torch_requirements": torch_requirements,
    "torch_index_url": torch_index_url,
    "env_preexisting": env_preexisting,
    "git_head": git_head,
    "verify": json.loads(verify_lines[-1]),
}
print(json.dumps(payload, ensure_ascii=False))
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or refresh the remote trajair conda environment.")
    add_remote_target_args(parser)
    add_remote_project_root_arg(parser)
    add_remote_conda_args(parser)
    parser.add_argument("--python-version", default="3.8.10")
    parser.add_argument("--requirements-file", default="requirements.txt")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu121")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the remote conda env.")
    return parser.parse_args()


def load_requirements(path: str) -> list[str]:
    rows: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line)
    return rows


def main() -> int:
    args = parse_args()
    requirements = load_requirements(args.requirements_file)
    env_parts = {
        "PROJECT_ROOT": args.project_root,
        "CONDA_SH": args.conda_sh,
        "CONDA_ENV": args.conda_env,
        "PYTHON_VERSION": args.python_version,
        "REQUIREMENTS_JSON": json.dumps(requirements, ensure_ascii=False),
        "TORCH_INDEX_URL": args.torch_index_url,
    }
    if args.recreate:
        env_parts["RECREATE_ENV"] = "1"
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_parts.items())
    command = f"{env_prefix} python3 - <<'PY'\n{REMOTE_SETUP_SCRIPT}\nPY"

    try:
        stdout = run_target_command(
            args.host,
            args.project_root,
            args.user,
            args.port,
            command,
            password_env=args.password_env,
            timeout=3600,
        )
    except Exception as exc:
        print("远端环境配置失败。", file=sys.stderr)
        if getattr(exc, "stdout", ""):
            print(exc.stdout, file=sys.stderr)
        if getattr(exc, "stderr", ""):
            print(exc.stderr, file=sys.stderr)
        if not getattr(exc, "stdout", "") and not getattr(exc, "stderr", ""):
            print(str(exc), file=sys.stderr)
        return 1

    payload = json.loads(stdout)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
