import sdflow


def test_version() -> None:
    from importlib.metadata import version

    assert sdflow.__version__ == version("sdflow")
