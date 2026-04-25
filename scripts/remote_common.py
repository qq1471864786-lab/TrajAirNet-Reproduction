from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import socket
import subprocess
from pathlib import Path

try:
    import paramiko
except Exception:  # pragma: no cover - optional dependency
    paramiko = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DEFAULT_REMOTE_HOST = os.environ.get("PROTOBASIS_REMOTE_HOST", "10.252.18.70")
DEFAULT_REMOTE_PORT = _env_int("PROTOBASIS_REMOTE_PORT", 30773)
DEFAULT_REMOTE_USER = os.environ.get("PROTOBASIS_REMOTE_USER", "root")
DEFAULT_REMOTE_PROJECT_ROOT = os.environ.get("PROTOBASIS_REMOTE_PROJECT_ROOT", "/3250604003/ProtoBasis-Net")
DEFAULT_REMOTE_CONDA_SH = os.environ.get(
    "PROTOBASIS_REMOTE_CONDA_SH",
    "/root/miniconda/etc/profile.d/conda.sh",
)
DEFAULT_REMOTE_CONDA_ENV = os.environ.get("PROTOBASIS_REMOTE_CONDA_ENV", "trajair")
DEFAULT_REMOTE_PASSWORD_ENV = os.environ.get("PROTOBASIS_REMOTE_PASSWORD_ENV", "PROTOBASIS_REMOTE_PASSWORD")


def add_remote_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--user", default=DEFAULT_REMOTE_USER)
    parser.add_argument(
        "--password-env",
        default=DEFAULT_REMOTE_PASSWORD_ENV,
        help=(
            "Environment variable holding the SSH password. "
            "Leave empty to use ssh/scp key-based auth."
        ),
    )


def add_remote_project_root_arg(
    parser: argparse.ArgumentParser,
    *,
    flag: str = "--project-root",
    default: str = DEFAULT_REMOTE_PROJECT_ROOT,
) -> None:
    parser.add_argument(flag, default=default)


def add_remote_conda_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--conda-sh", default=DEFAULT_REMOTE_CONDA_SH)
    parser.add_argument("--conda-env", default=DEFAULT_REMOTE_CONDA_ENV)


def ssh_target(user: str, host: str) -> str:
    return f"{user}@{host}"


def get_password(password_env: str) -> str | None:
    if not password_env:
        return None
    password = os.environ.get(password_env, "").strip()
    return password or None


def local_addresses() -> set[str]:
    addresses = {"127.0.0.1", "::1", "localhost"}
    for name in {socket.gethostname(), socket.getfqdn()}:
        if name:
            addresses.add(name.lower())
        try:
            for _, _, _, _, sockaddr in socket.getaddrinfo(name, None):
                if sockaddr and sockaddr[0]:
                    addresses.add(sockaddr[0].lower())
        except OSError:
            continue
    return addresses


def should_run_locally(host: str, project_root: str) -> bool:
    if not Path(project_root).exists():
        return False

    host_norm = host.strip().lower()
    local_addrs = local_addresses()
    if host_norm in local_addrs:
        return True

    try:
        for _, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            if sockaddr and sockaddr[0].lower() in local_addrs:
                return True
    except OSError:
        return False
    return False


def run_local(command: str) -> str:
    result = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _paramiko_client(
    *,
    user: str,
    host: str,
    port: int,
    password_env: str,
    timeout: int | float | None = 60,
):
    if paramiko is None:
        raise RuntimeError(
            f"需要 paramiko 才能通过 {password_env} 使用密码认证，但当前环境未安装 paramiko。"
        )

    password = get_password(password_env)
    if not password:
        return None

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        timeout=timeout,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run_ssh(
    user: str,
    host: str,
    port: int,
    command: str,
    *,
    password_env: str = DEFAULT_REMOTE_PASSWORD_ENV,
    timeout: int | float | None = 60,
) -> str:
    client = _paramiko_client(
        user=user,
        host=host,
        port=port,
        password_env=password_env,
        timeout=timeout,
    )
    if client is not None:
        try:
            _, stdout, stderr = client.exec_command(command, timeout=timeout)
            out_text = stdout.read().decode("utf-8", errors="replace")
            err_text = stderr.read().decode("utf-8", errors="replace")
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                raise subprocess.CalledProcessError(
                    exit_code,
                    command,
                    output=out_text,
                    stderr=err_text,
                )
            return out_text
        finally:
            client.close()

    result = subprocess.run(
        ["ssh", "-p", str(port), ssh_target(user, host), command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def run_target_command(
    host: str,
    project_root: str,
    user: str,
    port: int,
    command: str,
    *,
    password_env: str = DEFAULT_REMOTE_PASSWORD_ENV,
    timeout: int | float | None = 60,
) -> str:
    if should_run_locally(host, project_root):
        return run_local(command)
    return run_ssh(
        user,
        host,
        port,
        command,
        password_env=password_env,
        timeout=timeout,
    )


def _sftp_mkdirs(sftp, remote_dir: str) -> None:
    if not remote_dir:
        return

    current = "/" if remote_dir.startswith("/") else ""
    for part in [item for item in remote_dir.split("/") if item]:
        current = posixpath.join(current, part) if current else part
        try:
            sftp.stat(current)
        except IOError:
            sftp.mkdir(current)


def upload_file(
    local_path: str | Path,
    remote_path: str,
    *,
    user: str,
    host: str,
    port: int,
    password_env: str = DEFAULT_REMOTE_PASSWORD_ENV,
    timeout: int | float | None = 60,
) -> None:
    local_path = Path(local_path)
    client = _paramiko_client(
        user=user,
        host=host,
        port=port,
        password_env=password_env,
        timeout=timeout,
    )
    if client is not None:
        try:
            sftp = client.open_sftp()
            try:
                _sftp_mkdirs(sftp, posixpath.dirname(remote_path))
                sftp.put(str(local_path), remote_path)
            finally:
                sftp.close()
        finally:
            client.close()
        return

    remote_dir = posixpath.dirname(remote_path)
    if remote_dir:
        run_ssh(
            user,
            host,
            port,
            f"mkdir -p {shlex.quote(remote_dir)}",
            password_env=password_env,
            timeout=timeout,
        )
    subprocess.run(
        ["scp", "-P", str(port), str(local_path), f"{ssh_target(user, host)}:{remote_path}"],
        check=True,
    )
