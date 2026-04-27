from ramp import app


def test_smoke() -> None:
    assert app is not None
