from __future__ import annotations

import sys

from streamlit.web import cli as stcli


def build_streamlit_argv(extra_args: list[str] | None = None) -> list[str]:
    return ["streamlit", "run", "app.py", *(extra_args or [])]


def main(argv: list[str] | None = None) -> int:
    sys.argv = build_streamlit_argv(argv if argv is not None else sys.argv[1:])
    return int(stcli.main())


if __name__ == "__main__":
    raise SystemExit(main())
