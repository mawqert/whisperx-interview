#!/bin/sh
set -eu
cd -- "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

if [ ! -x .venv/bin/python ]; then
    printf '%s\n' 'Missing virtual environment. Run:' \
        '  python3 -m venv .venv' \
        '  .venv/bin/pip install -r requirements.txt' >&2
    exit 1
fi

exec .venv/bin/python app.py
