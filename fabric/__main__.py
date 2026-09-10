"""Allow ``python -m fabric`` as well as the ``fabric`` console script."""

from fabric.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
