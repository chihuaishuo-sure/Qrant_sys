#!/usr/bin/env bash
# macOS / Linux 启动脚本
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then python3 -m venv .venv; fi
if [ ! -f .venv/deps_ok.txt ]; then
  .venv/bin/python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || .venv/bin/python -m pip install -r requirements.txt
  echo ok > .venv/deps_ok.txt
fi
exec .venv/bin/python -m streamlit run app.py --server.port 8501
