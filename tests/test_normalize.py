import pytest

from wyoming_nemo_speech.normalize import sentences, spoken


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello. How can I assist?", "Hello. How can I assist?"),
        ("27°", "twenty-seven degrees"),
        (
            "It is 68.5°F outside.",
            "It is sixty-eight point five degrees Fahrenheit outside.",
        ),
        ("-5 °C", "minus five degrees Celsius"),
        ("Humidity is 72%.", "Humidity is seventy-two percent."),
        ("at 7:30 PM", "at seven thirty PM"),
        ("at 7:05", "at seven oh five"),
        ("at 12:00", "at twelve"),
        ("the 2nd light and the 21st", "the second light and the twenty-first"),
        ("1,234 steps", "one thousand two hundred thirty-four steps"),
        ("3 lights", "three lights"),
        ("version 0.5", "version zero point five"),
    ],
)
def test_spoken(text: str, expected: str) -> None:
    assert spoken(text) == expected


@pytest.mark.parametrize(
    ("text", "complete", "rest"),
    [
        ("Hello again there. How are", ["Hello again there."], " How are"),
        ("The light is on! Is that all? ", ["The light is on!", "Is that all?"], " "),
        (
            "First line here\nSecond line here\n",
            ["First line here", "Second line here"],
            "",
        ),
        ("It is 3.5 degrees out. Next one", ["It is 3.5 degrees out."], " Next one"),
        ("Dr. Smith is here. Next one", ["Dr. Smith is here."], " Next one"),
        ("OK. The light is on. Next", ["OK. The light is on."], " Next"),
        ('He said "Stop here." Then left', ['He said "Stop here."'], " Then left"),
        ("No boundary yet", [], "No boundary yet"),
    ],
)
def test_sentences(text: str, complete: list[str], rest: str) -> None:
    assert sentences(text) == (complete, rest)
