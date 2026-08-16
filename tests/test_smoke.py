"""The package imports and the toolchain runs."""


def test_package_imports() -> None:
    import footy

    assert footy is not None
