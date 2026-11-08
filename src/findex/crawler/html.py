"""HTML -> (title, visible text, links), with the standard library's ``html.parser``."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})
BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "pre",
        "section",
        "article",
        "header",
        "footer",
        "dd",
        "dt",
    }
)


@dataclass(frozen=True, slots=True)
class ParsedPage:
    title: str
    text: str
    links: tuple[str, ...]


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in BLOCK_TAGS:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        elif not self._skip:
            self.text_parts.append(data)


def parse_html(html: str) -> ParsedPage:
    p = _Extractor()
    p.feed(html)
    p.close()
    text = "\n".join(line.strip() for line in "".join(p.text_parts).splitlines())
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return ParsedPage(" ".join("".join(p.title_parts).split()), text.strip(), tuple(p.links))
