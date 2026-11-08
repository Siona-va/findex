"""``python -m findex ...`` == the ``findex`` command (handy for profilers: Lab 8)."""

from findex.cli import app

if __name__ == "__main__":
    app(prog_name="findex")
