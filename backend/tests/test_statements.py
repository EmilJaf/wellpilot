"""Answers as statements printed from the tool results (design D): references, printing,
what is refused, and a scripted run through the loop. No tokens are spent."""

import asyncio
import json
from datetime import date
from pathlib import Path

import pytest
from test_agent import completion, fake_llm

from wellpilot.agent.evidence import EvidenceLog
from wellpilot.agent.loop import SUBMIT, run_agent, system_prompt
from wellpilot.agent.statements import (
    BadRef,
    StatementSubmission,
    fmt,
    render,
    resolve,
)
from wellpilot.data.store import ProductionStore

FIXTURES = Path(__file__).parent / "fixtures"
REGIME = "E5[kind=regime_change,signal=oil_rate]/details"


@pytest.fixture(scope="module")
def log() -> EvidenceLog:
    """The 2010 time machine's results (run 99d66f7a3adb, as of 2010-12-31)."""
    run = json.loads((FIXTURES / "runs" / "99d66f7a3adb.json").read_text())
    return EvidenceLog.restore(run["evidence"].values())


def printed(log, *findings, summary=None, confidence="fact"):
    answer = {
        "summary": summary or [{"say": "the answer"}],
        "findings": [{"parts": parts, "confidence": confidence} for parts in findings],
    }
    return render(StatementSubmission.model_validate(answer), log)


def test_a_value_prints_with_its_own_labels(log):
    out = printed(log, [{"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"}])
    assert out.problems == []
    # The screen's water cut is a median over its window: the label says so, not the model.
    assert out.texts[1] == ("F-12's median water cut over the 90 days to 2010-12-31 was 69%.")
    (number,) = out.submission.findings[0].numbers
    assert (number.text, number.value, number.evidence_id) == ("69%", 0.6939, "E1")


def test_a_wrong_cell_prints_a_true_sentence_about_that_cell(log):
    """The attack the checker could not stop in prose: the model copies another row. Here
    the sentence names the row it took, so it stays true."""
    right = printed(log, [{"kind": "value", "ref": "E7/monthly[month=2010-12]/water_cut"}])
    wrong = printed(log, [{"kind": "value", "ref": "E7/monthly[month=2010-06]/water_cut"}])
    assert "in 2010-12" in right.texts[1] and "in 2010-06" in wrong.texts[1]
    assert right.texts[1] != wrong.texts[1]


def test_a_regime_change_prints_as_averages_and_the_choke_across_it(log):
    out = printed(
        log,
        [
            {"kind": "change", "ref": f"{REGIME}/before_mean", "to": f"{REGIME}/after_mean"},
            {"say": ", while"},
            {
                "kind": "change",
                "ref": f"{REGIME}/choke_before_pct",
                "to": f"{REGIME}/choke_after_pct",
            },
        ],
    )
    assert out.problems == []
    # Six numbers in one claim: the oil rate's averages move to the detail.
    assert out.texts[1] == (
        "F-12's oil rate averaged 55% less after 2010-05-02 than before, while its choke "
        "opening fell from 83% to 58% over the 4 weeks either side of 2010-05-02."
    )
    assert "4,207 Sm3/d before and 1,888 Sm3/d after" in out.submission.findings[0].detail
    alone = printed(
        log, [{"kind": "change", "ref": f"{REGIME}/before_mean", "to": f"{REGIME}/after_mean"}]
    )
    assert alone.texts[1] == (
        "F-12's oil rate averaged 55% less after 2010-05-02 than before (4,207 Sm3/d before, "
        "1,888 Sm3/d after)."
    )


def test_a_change_between_rows_and_a_comparison(log):
    out = printed(
        log,
        [
            {
                "kind": "change",
                "ref": "E7/monthly[month=2009-10]/water_cut",
                "to": "E7/monthly[month=2010-12]/water_cut",
            }
        ],
        [
            {
                "kind": "compare",
                "ref": "E1/wells[well=F-12]/oil_rate_sm3_per_day",
                "against": "E1/wells[well=F-14]/oil_rate_sm3_per_day",
            }
        ],
    )
    assert out.problems == []
    assert out.texts[1] == "F-12's water cut rose from 5.1% in 2009-10 to 71% in 2010-12."
    assert "is 71% of F-14's median oil rate" in out.texts[2]


def test_events_forecasts_and_quotes(log):
    out = printed(
        log,
        [{"kind": "event", "ref": "E5[kind=gauge_failure]"}],
        [{"kind": "forecast", "ref": "E13"}],
        confidence="inference",
    )
    assert out.problems == []
    assert out.texts[1] == "F-12's downhole gauge read zero from 2010-10-12 to 2010-12-31."
    assert out.texts[2] == (
        "F-12 is forecast to make about 359,000 Sm3 of oil from 2011-01-01 to 2011-12-31, "
        "range 324,000 to 470,000 Sm3 from the errors of 5 earlier forecasts made the same "
        "way."
    )


def test_a_quote_is_the_fact_s_exact_words():
    run = json.loads((FIXTURES / "runs" / "43c53d5c1096.json").read_text())
    log = EvidenceLog.restore(run["evidence"].values())
    reports = next(e for e in log if e.tool == "search_reports")
    out = printed(log, [{"kind": "quote", "ref": f"{reports.id}/facts[0]"}])
    assert out.problems == []
    assert out.texts[1].startswith("The development plan says “")


@pytest.mark.parametrize(
    "ref, says",
    [
        ("E7/monthly[month=2015-13]/water_cut", "no row of E7/monthly has month=2015-13"),
        ("E1/wells[well=F-99]/water_cut", "its well values: F-12, F-14"),
        ("E1/wells[well=F-12]/watercut", "has no field 'watercut'; its fields: well, role"),
        ("E5[kind=regime_change]", "2 rows of E5 match [kind=regime_change]"),
        ("E1/wells[well=F-12]", "is a record, not a number"),
        ("E2/monthly", "E2 is a table (it has no field 'monthly'): pick its row first"),
        ("E2/first_oil", "e.g. 'E2[well=F-12]/first_oil'"),
        ("E99/x", "there is no evidence E99"),
        ("F-12 water cut", "a reference starts with an evidence id"),
    ],
)
def test_a_bad_reference_says_what_is_there(log, ref, says):
    with pytest.raises(BadRef, match=None) as caught:
        cell = resolve(ref, log)
        from wellpilot.agent.statements import number

        number(cell)
    assert says in str(caught.value)


def test_free_words_hold_nothing_a_reference_should(log):
    value = {"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"}
    for words, what in [
        ("about 70", "a digit"),
        ("since October", "a month name"),
        ("in barrels", "a unit"),
        ("nearly half", "an amount in words"),
        ("“maksimum vannkutt”", "a quote mark"),
        ("as F-14 did", "none of its statements is about F-14"),
    ]:
        out = printed(log, [value, {"say": words}])
        assert any(what in p for p in out.problems), (words, out.problems)
    assert printed(log, [value, {"say": ", which may well be the gauge"}]).problems == []


def test_free_words_may_name_the_well_their_statements_are_about(log):
    """The trial run (9ca1f130efea) lost every well name to the digit rule and wrote
    headings like 'downhole gauge reads zero'. A name its statements back is allowed, and
    words that start with it make the next statement say 'its'."""
    out = printed(
        log,
        [
            {"say": "F-12 needs attention most"},
            {"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"},
            {"say": "and"},
            {"kind": "value", "ref": "E1/wells[well=F-14]/water_cut"},
        ],
    )
    assert out.problems == []
    assert out.texts[1] == (
        "F-12 needs attention most: its median water cut over the 90 days to 2010-12-31 was "
        "69% and F-14's median water cut over the same 90 days was 61%."
    )
    # Two wells in the words: the statement names its own.
    both = printed(
        log,
        [
            {"say": "F-12 and F-14 both water out"},
            {"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"},
        ],
    )
    assert both.texts[1].startswith("F-12 and F-14 both water out: F-12's median")


@pytest.mark.parametrize(
    "parts, text",
    [
        # A statement after a statement, then words that do not continue it.
        (
            [
                {"kind": "event", "ref": "E5[kind=gauge_failure]"},
                {"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"},
                {"say": "even as the well kept flowing"},
            ],
            "F-12's downhole gauge read zero from 2010-10-12 to 2010-12-31; its median water "
            "cut over the 90 days to 2010-12-31 was 69%, even as the well kept flowing.",
        ),
        # Words that lead into a statement take no colon; the model's own punctuation stays.
        (
            [
                {"say": "the gauge is dead."},
                {"kind": "event", "ref": "E5[kind=gauge_failure]"},
                {"say": "while"},
                {"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"},
            ],
            "The gauge is dead. F-12's downhole gauge read zero from 2010-10-12 to "
            "2010-12-31, while its median water cut over the 90 days to 2010-12-31 was 69%.",
        ),
        # "than before" ends a phrase: the statement after it is a new clause.
        (
            [
                {"say": "water came in faster than before"},
                {"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"},
            ],
            "Water came in faster than before: F-12's median water cut over the 90 days to "
            "2010-12-31 was 69%.",
        ),
    ],
)
def test_free_words_and_statements_join_into_sentences(log, parts, text):
    out = printed(log, parts)
    assert out.problems == [] and out.texts[1] == text


def test_a_plug_s_windows_print_as_the_job_and_the_return_not_one_day():
    """production_log's plugs_set: the windows are the 60 days before the job and the first
    60 back on stream. The trial printed both ends 'on 2010-09-23'."""
    run = json.loads((FIXTURES / "runs" / "83a3114d185f.json").read_text())
    log = EvidenceLog.restore(run["evidence"].values())
    plt = next(e for e in log if e.tool == "production_log" and e.result.get("plugs_set"))
    plug = plt.result["plugs_set"][0]
    out = printed(
        log,
        [
            {
                "kind": "change",
                "ref": f"{plt.id}/plugs_set[0]/before/water_cut",
                "to": f"{plt.id}/plugs_set[0]/after/water_cut",
            },
            {"say": ", while"},
            {
                "kind": "change",
                "ref": f"{plt.id}/plugs_set[0]/before/oil_sm3_per_day",
                "to": f"{plt.id}/plugs_set[0]/after/oil_sm3_per_day",
            },
        ],
    )
    assert out.problems == []
    claim = out.submission.findings[0].claim
    depth = f"{plug['depth_m']:,.0f}"
    assert f"before the job that set the plug at {depth} m on {plug['date']}" in claim
    assert "the first 60 days back on stream" in claim and "over the same days" in claim
    assert claim.count(plug["date"]) == 1
    declared = {n.text: n.value for n in out.submission.findings[0].numbers}
    assert declared[depth] == plug["depth_m"]


def test_a_claim_with_too_many_numbers_moves_values_to_its_detail(log):
    """Two regime-change statements print five numbers; the claim keeps the percentage
    and the choke, the oil rate's averages go to the detail (still printed by code)."""
    out = printed(
        log,
        [
            {"kind": "change", "ref": f"{REGIME}/before_mean", "to": f"{REGIME}/after_mean"},
            {"say": ", while"},
            {
                "kind": "change",
                "ref": f"{REGIME}/choke_before_pct",
                "to": f"{REGIME}/choke_after_pct",
            },
            {"say": "and"},
            {"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"},
        ],
    )
    finding = out.submission.findings[0]
    assert out.problems == []
    assert "4,207" not in finding.claim and "averaged 55% less after 2010-05-02" in finding.claim
    assert finding.detail == (
        "F-12 oil rate: 4,207 Sm3/d before and 1,888 Sm3/d after 2010-05-02, the periods' averages."
    )
    assert {n.text for n in finding.numbers} == {"83%", "58%", "69%"}


def test_a_fact_cannot_rest_on_warnings_a_forecast_or_a_cause(log):
    fit = printed(log, [{"kind": "value", "ref": "E11/best/di_per_year"}])
    assert "rests on E11, which carries warnings" in fit.problems[0]
    run = json.loads((FIXTURES / "runs" / "99d66f7a3adb.json").read_text())
    run["evidence"]["E13"]["warnings"] = []  # a sound fit: still a forecast
    sound = EvidenceLog.restore(run["evidence"].values())
    forecast = printed(sound, [{"kind": "forecast", "ref": "E13"}])
    assert forecast.problems == [
        "Finding 1 is marked fact but holds a forecast: mark it inference."
    ]
    cause = printed(
        log,
        [{"kind": "event", "ref": "E5[kind=gauge_failure]"}, {"say": "because of the shut-in"}],
    )
    assert "states a cause ('because')" in cause.problems[0]
    hedged = printed(
        log,
        [{"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"}],
        summary=[{"say": "water likely drove the fall"}],
    )
    assert hedged.problems == []


def test_a_change_must_stay_within_one_quantity_and_one_well(log):
    across = printed(
        log,
        [
            {
                "kind": "change",
                "ref": "E1/wells[well=F-12]/water_cut",
                "to": "E1/wells[well=F-12]/uptime",
            }
        ],
    )
    assert "are not the same quantity" in across.problems[0]
    wells = printed(
        log,
        [
            {
                "kind": "change",
                "ref": "E1/wells[well=F-12]/water_cut",
                "to": "E1/wells[well=F-14]/water_cut",
            }
        ],
    )
    assert "F-12's" in wells.problems[0] and "F-14's" in wells.problems[0]


@pytest.mark.parametrize(
    "value, unit, text",
    [
        (0.6939, "share", "69%"),
        (0.0512, "share", "5.1%"),
        (359007, "Sm3", "359,000 Sm3"),
        (1551, "rate", "1,551 Sm3/d"),
        (3569815, "Sm3", "3.57 million Sm3"),
        (1.31, "per_year", "1.31 per year"),
        (-0.5512, "change", "55%"),
    ],
)
def test_numbers_print_with_three_significant_digits_at_most(value, unit, text):
    assert fmt(value, unit) == text


def test_the_prompt_asks_for_statements_and_drops_the_declarations():
    store = ProductionStore(FIXTURES / "processed", as_of=date(2010, 12, 31))
    prose, statements = system_prompt(store), system_prompt(store, statements=True)
    assert "Declare every number" in prose and "Declare every number" not in statements
    assert "Write the answer as statements" in statements
    assert statements.endswith("Data is available only up to 2010-12-31.")


def test_a_scripted_run_is_sent_back_then_verified():
    """Through the loop: a bad reference costs a round with a message the model can act on,
    and the printed answer stands in for a prose one (the UI reads it from the step)."""
    store = ProductionStore(FIXTURES / "processed", as_of=date(2010, 12, 31))
    bad = {
        "summary": [{"kind": "value", "ref": "E1/wells[well=F-12]/water_cut"}],
        "findings": [
            {
                "parts": [{"kind": "value", "ref": "E1/wells[well=F-12]/watercut"}],
                "confidence": "fact",
            }
        ],
    }
    good = json.loads(json.dumps(bad).replace("/watercut", "/water_cut"))
    llm, completions = fake_llm(
        [
            completion(tool_calls=[("field_screen", {})]),
            completion(tool_calls=[(SUBMIT, bad)]),
            completion(tool_calls=[(SUBMIT, good)]),
        ]
    )
    events = []
    run = asyncio.run(run_agent("Which well?", llm, store, statements=True, on_event=events.append))
    assert run.verified and len(run.check_rounds) == 2
    assert "has no field 'watercut'" in run.check_rounds[0][0]
    assert run.findings[0].claim == (
        "F-12's median water cut over the 90 days to 2010-12-31 was 69%."
    )
    assert run.findings[0].numbers[0].evidence_id == "E1"
    submit_tool = next(
        t for t in completions.requests[0]["tools"] if t["function"]["name"] == SUBMIT
    )
    assert "parts" in json.dumps(submit_tool)
    sent_back = json.loads(completions.requests[2]["messages"][-1]["content"])
    assert sent_back["printed"] and "_rendered" not in sent_back
    shown = [c for e in events if e["type"] == "step" for c in e["tool_calls"]]
    final = [c for c in shown if c["name"] == SUBMIT][-1]
    assert final["arguments"]["findings"][0]["claim"] == run.findings[0].claim
    assert final["statements"] == good
