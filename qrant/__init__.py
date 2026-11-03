"""Qrant —— 先求少亏的个人量化风控系统。"""

__version__ = "1.2.0"


def _use_venv() -> None:
    """用系统 Python 直接运行脚本（没装依赖）时，自动改用项目的 .venv 重新运行。"""
    try:
        import pandas  # noqa: F401
        return
    except ImportError:
        pass
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    venv = next((p for p in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python")
                 if p.exists()), None)
    script = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if venv is not None and script is not None and script.suffix == ".py" \
            and Path(sys.executable).resolve() != venv.resolve():
        sys.exit(subprocess.run([str(venv), *sys.argv]).returncode)
    raise SystemExit("缺少依赖（pandas 等）。请先双击 启动.bat 完成安装，"
                     "或运行：python -m pip install -r requirements.txt")


_use_venv()
