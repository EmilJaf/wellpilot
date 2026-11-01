"""False rejections from the eval of Oct 31 (the corpus pins the real cases; these pin each
rule beside what it must still stop)."""

from wellpilot.agent import checker as C


def describe(text: str, number: str) -> dict:
    text = C._normalize(text)
    blanked = C._blank(text, C.WELL, C.DATE, C.CALENDAR_WINDOW, C.REFERENCE)
    i = text.index(number)
    return C._description(text, blanked, i, i + len(number))


def test_at_the_same_time_is_when_not_an_unchanged_choke():
    assert not C.CHOKE_STILL.search("the choke was cut at the same time from 76% to 39%")
    assert C.CHOKE_STILL.search("while the choke stayed the same")
    assert C.CHOKE_STILL.search("the choke was the same before and after")


def test_the_end_of_a_date_range_dates_no_number():
    said = describe("a workover from 2014-11-22 to 2014-11-27 that set a plug at 3150 m", "3150")
    assert "dates" not in said
    said = describe("from 2011 to 2013 with a correlation of 69%", "69%")
    assert "dates" not in said
    # One date still dates its number.
    assert describe("on 2014-11-26 it set a plug at 3150 m", "3150")["dates"][0] == "2014-11-26"


def test_a_correlation_word_ends_a_names_reach():
    said = describe("water cut and choke moved together with a correlation of 69%", "69%")
    assert "quantity" not in said
    assert describe("the choke was 69% in 2013", "69%")["quantity"][0] == "choke"


def test_a_limit_word_before_and_is_the_earlier_numbers():
    text = "water cut of 67% under the 80% limit and a field liquid rate of 10,700 Sm3/d"
    assert "ref" not in describe(text, "10,700")
    assert describe(text, "80%")["ref"][0] == "limit"
    assert describe("below the design limit of 13,000 Sm3/d", "13,000")["ref"][0] == "limit"


def test_water_handling_with_a_hyphen_names_its_limit():
    import re

    assert re.search(C.LIMIT_NAMES["water_sm3_per_day"], "the water-handling capacity")
    assert re.search(C.LIMIT_NAMES["water_sm3_per_day"], "water handling stayed below")


def test_same_time_wells_own_no_number():
    text = "F-12 was shut in 32 days (+366% same time F-1 C F-15 D)."
    assert C._named(text, text.index("32")) == {C._well_key("F-12")}


def test_a_worst_day_is_a_rate_not_a_count_of_days():
    """Eval of Nov 1 (run 39a70f886711): "max day 18020 Sm3/d" ended the run unverified."""
    assert C._key_dim("max_day") == "rate"
    assert C._key_dim("max_day_water_sm3") == "rate"
    assert C._key_dim("days_at_or_above") == "days"
    assert C._key_dim("zero_days") == "days"
