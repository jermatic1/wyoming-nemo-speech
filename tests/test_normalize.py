import pytest

from wyoming_nemo_speech.normalize import spoken


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
