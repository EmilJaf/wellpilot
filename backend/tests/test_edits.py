"""A reader's edit of an answer: the agent's declared numbers carried to the edited text."""

from wellpilot.agent.checker import DeclaredNumber
from wellpilot.agent.edits import redeclare


def d(text: str, value: float, eid: str = "E1") -> DeclaredNumber:
    return DeclaredNumber(text=text, value=value, evidence_id=eid)


CLAIM = "F-12 made 1 551 Sm3/d at 69% water cut, 123 258 Sm3 since 2010-10-12."
NUMBERS = [d("1 551", 1551, "E1"), d("69%", 0.6939, "E3"), d("123 258", 123258, "E1")]


def test_an_untouched_number_keeps_its_declaration() -> None:
    edited = CLAIM.replace("water cut", "water cut (rising)")
    assert redeclare(CLAIM, NUMBERS, edited) == NUMBERS


def test_a_replaced_number_is_declared_as_written_with_the_same_source() -> None:
    edited = CLAIM.replace("1 551", "1 800")
    assert redeclare(CLAIM, NUMBERS, edited) == [d("1 800", 1800, "E1"), *NUMBERS[1:]]


def test_a_percent_keeps_its_scale_open() -> None:
    edited = CLAIM.replace("69%", "75%")
    assert redeclare(CLAIM, NUMBERS, edited)[1] == d("75%", 75, "E3")


def test_a_removed_number_drops_its_declaration() -> None:
    edited = CLAIM.replace(" at 69% water cut", "")
    assert redeclare(CLAIM, NUMBERS, edited) == [NUMBERS[0], NUMBERS[2]]


def test_a_number_typed_elsewhere_is_not_declared() -> None:
    edited = CLAIM + " The choke was 53%."
    assert redeclare(CLAIM, NUMBERS, edited) == NUMBERS


def test_text_before_or_after_a_number_does_not_change_it() -> None:
    edited = CLAIM.replace("1 551 Sm3/d", "about 1 551 Sm3/d of oil")
    assert redeclare(CLAIM, NUMBERS, edited) == NUMBERS


def test_the_norwegian_convention_follows_the_original() -> None:
    claim = "The plan's plateau is 9.000 Sm3/d."
    carried = redeclare(claim, [d("9.000", 9000, "E6")], claim.replace("9.000", "8.500"))
    assert carried == [d("8.500", 8500, "E6")]


def test_legacy_answers_keep_reading_numbers_from_the_text() -> None:
    assert redeclare(CLAIM, None, CLAIM.replace("1 551", "1 800")) is None


def test_a_declaration_the_text_never_wrote_is_kept() -> None:
    """Featured run e51eada95269 declared '100%' without writing it; the checker reads its
    "near 1.0" as that value in another form. Dropping it made the unedited claim fail."""
    numbers = [
        DeclaredNumber(text="0.98", value=0.98, evidence_id="E4"),
        DeclaredNumber(text="100%", value=100, evidence_id="E4"),
    ]
    text = "Injection kept pace with voidage near 1.0."

    assert redeclare(text, numbers, text) == numbers
    assert redeclare(text, numbers, text.replace("1.0", "1.2")) == numbers
