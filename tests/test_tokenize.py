"""Lab 1: plain assert-style checks (run with `python tests/test_tokenize.py` or pytest)."""

from findex.tokenize import tokenize


def toks(text: str) -> list[str]:
    return list(tokenize(text))


def test_mixed_case() -> None:
    assert toks("Hello WORLD hello") == ["hello", "world", "hello"]


def test_casefold_not_lower() -> None:
    assert toks("Straße") == toks("STRASSE") == ["strasse"]


def test_cyrillic() -> None:
    assert toks("Привіт, Світе! Їжак і ґанок") == ["привіт", "світе", "їжак", "і", "ґанок"]


def test_combining_mark_nfc() -> None:
    assert toks("café") == toks("café") == ["café"]


def test_punctuation() -> None:
    assert toks("Hi!!! (yes), no... ok?") == ["hi", "yes", "no", "ok"]


def test_empty_string() -> None:
    assert toks("") == []
    assert toks("   \n\t ... ") == []


def test_apostrophes_kept_and_unified() -> None:
    expected = ["don't", "п'ять", "п'ять", "п'ять", "quoted"]
    assert toks("don't п'ять п’ять пʼять 'quoted'") == expected


def test_hyphens_split() -> None:
    assert toks("free-threaded e-mail") == ["free", "threaded", "e", "mail"]


def test_digits() -> None:
    assert toks("Python 3.13 has 2x speed") == ["python", "3", "13", "has", "2x", "speed"]


def test_underscores() -> None:
    assert toks("__init__ sys_path ___") == ["__init__", "sys_path"]


def test_is_generator() -> None:
    import inspect

    assert inspect.isgeneratorfunction(tokenize)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
