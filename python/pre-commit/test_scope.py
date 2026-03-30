EXTRA_TRIGGERS: dict[str, list[str]] = {}

FULL_RUN_TRIGGERS: list[str] = [
    "pyproject.toml",
    "uv.lock",
    "pre-commit/pytest",
    "pre-commit/test_scope.py",
]
