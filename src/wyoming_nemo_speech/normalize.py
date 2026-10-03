"""Prepare text for Magpie: split into sentences, spell out numbers and symbols."""

from __future__ import annotations

import re
from typing import cast

import inflect

_inflect = inflect.engine()

_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_DEGREES = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*°\s*([CF])?")
_PERCENT = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*%")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

_SCALES = {"C": " Celsius", "F": " Fahrenheit"}

_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s)|\n")
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st", "vs", "etc", "e.g", "i.e"}
_MIN_WORDS = 3


def sentences(text: str) -> tuple[list[str], str]:
    """Split off complete sentences and return them with the unfinished rest.

    Fragments shorter than a few words are merged into the next sentence.
    """
    complete: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if match.group().startswith(".") and _ends_abbreviation(text[: match.start()]):
            continue
        candidate = " ".join(text[start : match.end()].split())
        if len(candidate.split()) < _MIN_WORDS:
            continue
        complete.append(candidate)
        start = match.end()
    return complete, text[start:]


def _ends_abbreviation(text: str) -> bool:
    words = text.split()
    word = words[-1].lstrip("\"'([") if words else ""
    initial = len(word) == 1 and word.isalpha()
    return initial or word.lower() in _ABBREVIATIONS


def spoken(text: str) -> str:
    text = _TIME.sub(_time, text)
    text = _ORDINAL.sub(lambda m: _words(_inflect.ordinal(_word(m[1]))), text)
    text = _DEGREES.sub(
        lambda m: f"{_words(m[1])} degrees{_SCALES.get(m[2] or '', '')}", text
    )
    text = _PERCENT.sub(lambda m: f"{_words(m[1])} percent", text)
    return _NUMBER.sub(lambda m: _words(m[0]), text)


def _words(number: str) -> str:
    word = _word(number.replace(",", ""))
    return str(_inflect.number_to_words(word, andword="")).replace(",", "")


def _word(text: str) -> inflect.Word:
    return cast(inflect.Word, text)


def _time(match: re.Match[str]) -> str:
    hour, minute = int(match[1]), int(match[2])
    if minute == 0:
        return _words(str(hour))
    if minute < 10:
        return f"{_words(str(hour))} oh {_words(str(minute))}"
    return f"{_words(str(hour))} {_words(str(minute))}"
