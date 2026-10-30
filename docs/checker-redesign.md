# The checker: how to get under 1% errors

Status: analysis, Oct 30; stage 1 of option D (the feasibility trial) built the same day,
behind a flag, and not yet run with the model (see "Stage 1, built" at the end).

## What counts as an error

- **False pass:** an answer that misstates its source is accepted (a number from the wrong
  row, a wrong date, another well's value, a rise written as a fall).
- **False rejection:** a true answer is sent back. In a live run it costs a correction
  round and, as the corpus shows, can push the model into a worse answer.

Both have to be under 1%, on answers nobody has seen before, written by the model or by a
reader trying to fool it.

## Where the checker stands (Oct 30)

| Measure | Value |
|---|---|
| Wrong-cell swaps accepted (`tests/checker_bench.py`, 2,862 swaps over 349 corpus answers) | 28% (53% on Oct 29 morning) |
| Simulated reviewers' false edits accepted (panels 8 and 9) | 31%, 35% |
| Meaning-keeping rewordings of accepted answers rejected | 0 of 458 |
| Submissions in the corpus whose review found a false rejection | 41 of 349 (12%), each fixed after it happened |
| False rejections among the last eval's new submissions | 6 of 31 |

The 0% of false rejections is measured on answers the checker had already accepted, so it is
an in-sample number. On new answers the tail is open: every eval so far has found new false
rejections, and each fix is a new rule that can misfire.

The trade-off is steep. Asking the description fit for a margin of 2 instead of 3 takes false
passes from 28% to 25% and false rejections of true rewordings from 0% to 28%.

The ceiling is in the text, not in the rules. Of the 790 swaps that still pass, about a third
have no word that tells the two values apart ("the choke was later opened to 53%": 53% was
the 2010 average, 100% the value meant). Most of the rest are told apart only by something
the words say about the rival value and nothing about the written one. No rule can read what
the sentence does not say.

**So no amount of rules over free prose gets to 1% and 1%.** The options below change what
the model writes.

## Options

Measured offline on the same 2,862 swaps and 1,254 true declarations, with today's reading
of the words (`_description`, `_fit_scores`).

| Option | What the model writes | False passes (swaps) | False rejections | Effort |
|---|---|---|---|---|
| A. More rules (today) | prose + declared numbers | 28%, falling slowly | 0% in-sample; new ones in every eval | each rule a few hours, risk grows |
| B. Pinned cells | prose + each number's cell path | 40.5% (words alone rarely contradict a wrong cell) | 3.7% (3.6% on rewordings) | 1 day |
| C. Pinned cells, words must identify them | as B, prose must name what tells the cell apart | 3.7% | 56% of today's true claims fail (they do not say enough) | 1 day + much longer prose |
| D. Statements rendered from cells | typed statements whose values are references; the text is printed from the cells | 0 by construction (see below) | 0 from judgment; only format errors with exact messages | about 3 days + re-records |
| E. LLM or NLI judge | prose | see the literature below | see below | 1 day, tokens per check |
| F. Prose with inline references | prose with `{ref}` slots filled by the renderer, words checked against each cell | between B and D: bounded by how well the words are read | as B | 2 days |

**Why B and C fail.** Pinning a number to its cell does not help when the sentence says
nothing about the cell: 40% of wrong pins contradict no word. Requiring the words to identify
the cell closes that (3.7%), but 56% of the true claims in the corpus do not identify their
cell either, so they would all be sent back. People and models write "water cut fell to
31%", not "the monthly water cut of F-12 in 2015-01 was 31%".

**Why D works.** In D the model never types a number, a date, a well name or a unit. It
picks cells and a statement type; a renderer prints the sentence from the cells. If it picks
the wrong cell, the sentence is still true, about that cell:

```
change(from=E2[month=2014-10]/water_cut, to=E2[month=2015-01]/water_cut, note="after the late-November workover")
-> F-12's water cut fell from 96.7% in 2014-10 to 31.1% in 2015-01, after the late-November workover

the wrong-cell attack, to=E2[month=2015-09]/water_cut
-> F-12's water cut fell from 96.7% in 2014-10 to 73.7% in 2015-09, after the late-November workover
```

The second sentence is true; whether it is the right month to quote is a question of
selection (below), not of truth. In the simulation, only 4% of swaps landed on a cell with
the same well, quantity, time, side, statistic and grain as the true one, and all of them
were either numbers inside a tool's prose (109) or a calculation's inputs and two passes of
one survey (19). With full paths and no prose as a source, the residue is zero.

The prototype resolves paths into today's tool results as they are (column tables by row
key, lists of records by fields, nested fields), refuses what does not exist ("2015-13",
"F-99", a misspelt field) and prints three significant digits with units.

**Do today's answers fit a closed set of statement types?** Of the 1,309 numbers declared in
the corpus's accepted claims: 82% are exact cells and 3% rounded cells (both become
references), 8% are calculations (become relations the checker computes: change, ratio,
share, difference), 6% sit in a tool's prose or a report quote (quotes stay as verified fact
ids; tool prose repeats structured fields, which become the source), 0.2% were not found.
Their roles: a level 62%, the two ends of a change 23%, a duration or count 8%, a share 6%,
a percentage change 1.5%, a ratio 0.5%. About ten statement types cover them:

| Type | Example of what it renders |
|---|---|
| level | F-12's water cut was 69% on 2010-12-31 |
| change (verb and % computed) | F-12's oil rate fell 55% from 4,210 to 1,890 Sm3/d around 2010-05-02 |
| compare (ratio or difference computed) | F-12 made 4.8 times its rate before the job |
| share (computed) | F-12 made 46% of the field's oil |
| event | F-12's downhole gauge reads zero from 2010-10-12 |
| job effect (the tool's own windows) | after the 2014-11-22..27 workover, water cut 97% -> 34% over 60 days |
| limit (above or below computed) | the field's water cut first passed the plan's 0.80 in 2012-06 |
| forecast | about 359,000 Sm3 in 2011, range 324,000 to 470,000 from past errors |
| quote | "Maksimum vannkutt 0,80" (maximum water cut 0.80), PUD p.23 |
| duration or count | 27 months at or above the limit |

Free text joins the statements ("F-12 needs attention most:", "while", "so a new log should
come first"). It may hold no digits, dates, month names, well names or units: those can only
come from references. A cause or a recommendation is free text and stays an inference with a
hedge, as today.

## What D still cannot check

- **Selection:** a true statement about the wrong month, the wrong job, a weaker comparison.
  Partly closed by types that fix their own windows (a job's effect uses the tool's 60 days
  before and after, not months the model picks); the rest is the second opinion's and the
  reader's job.
- **Meaning in free words:** causes, recommendations, "the plan was optimistic". Labelled
  inference and hedged, as today; no method checks them deterministically.
- **Bugs in the renderer and resolver:** testable exhaustively, unlike prose: every numeric
  field of every tool's result type rendered and read back.

## The literature (research brief, Oct 30)

No paper found reports false accepts and false rejects separately for numeric claims. The
best figures by family:

| Family | Best reported | Error |
|---|---|---|
| Table claim verification: TabFact ([Ye 2023](https://arxiv.org/abs/2301.13808)), InfoTabs under word edits ([Gupta 2020](https://arxiv.org/abs/2005.06117)), numerical claims, QuanTemp ([Venktesh 2024](https://arxiv.org/abs/2403.17169)) | 93.0% (humans 92.1%); 65.6%; 58.3 F1 | 7-40% |
| LLM and NLI consistency checkers: LLM-AggreFact ([Tang 2024](https://arxiv.org/abs/2404.10774)), SummEdits ([Laban 2023](https://arxiv.org/abs/2305.14540)) | 75-80% balanced accuracy; 82.4% (humans about 91%) | 20-30% |
| Generation with citations: ALCE ([Gao 2023](https://arxiv.org/abs/2305.14627)), AttributionBench ([Li 2024](https://arxiv.org/abs/2402.15089)) | half not fully supported; about 80% F1 judging support | far |
| Neural data-to-text: RotoWire ([Puduppully 2019](https://arxiv.org/abs/1809.00582)), ToTTo ([Parikh 2020](https://arxiv.org/abs/2004.14373)) | 74.8-87.5% of facts right; 76.2% faithful | 12-25% |
| A neural model writing text from a correct logical form: Logic2Text ([Chen 2020](https://arxiv.org/abs/2004.14579)), weather MRs with constrained decoding ([Balakrishnan 2019](https://arxiv.org/abs/1906.07220)) | 82.4%; 91.8% | 8-18% |
| Templates bound to the data (RotoWire's template baseline) | 99.9% | under 1%, within the schema |
| Constrained decoding: structured outputs ([OpenAI 2024](https://openai.com/index/introducing-structured-outputs-in-the-api/)), PICARD ([Scholak 2021](https://arxiv.org/abs/2109.05093)) | 100% valid; invalid SQL 12% -> 2% | validity, not correctness |

What follows for us:
- **Anything that reads free prose (option E, and A at its best) stays at 7-30% error.**
  Humans disagree with each other 8-16% of the time on these tasks. An LLM judge is a second
  opinion (we have one), not a checker.
- **Only values bound to the data and text printed by deterministic code get near zero**,
  and only for what the schema can say. Even a model writing from a correct logical form
  makes 8-18% errors, so the renderer in D must be code, not the model.
- **A list of valid paths makes references valid, not right**: a wrong but existing cell is
  still possible. That is why D prints the cell's own labels, so a wrong pick is a true
  sentence about another cell, never a false one.
- **Strict formats can cost reasoning** ([Tam 2024](https://arxiv.org/abs/2408.02442)): the
  agent reasons and calls tools in free form as now, and only its final submission is typed.
- **Industry does the same:** Power BI smart narratives bind values in an author's text to
  data measures; Tableau Pulse computes its facts with deterministic models and lets a
  language model only phrase them; SEC Inline XBRL tags every value in a filing. None
  publishes an error rate.
- **Nothing checks** whether the cited cell is the relevant one, causes, comparisons and
  quantifiers in words, what was left out, or whether the tools themselves are right.

The full brief with the source caveats is in the session scratchpad (`brief.md`).

## Recommendation and plan

D, the way Power BI binds values into a narrative, but with each value's labels printed by
code too. It is the only family the literature and our simulation both put under 1% on both
sides, and only within its schema. Staged so that it can be dropped if the model cannot
write it:

1. **Feasibility, half a day.** Resolver, renderer and validator for the ten types in the
   backend, the submit schema v2 behind a flag, a prompt with examples. One live run of the
   time-machine question on Muse Spark Open (about $0.3, Emil runs it). Go if the model
   writes valid statements in two rounds or fewer.
2. **Backend, one day.** Every numeric field of every tool result addressable (selectors for
   events, jobs, survey passes; no numbers taken from tool prose). Today 79 of the corpus's
   declared numbers exist only in text: 51 in report facts and page text (the plan's 9,000
   and 0,80: these become a document-figure reference, the number as the page writes it with
   its exact quote), 21 in well-work summaries and event summaries (the 2014 plug's 3150 m
   is only in the daily report's prose: `well_operations` needs parsed plug and packer
   depths beside its perforation table), 5 in screen issue texts; warnings inherited from
   cells, so fact or inference follows from the references; precise messages for a bad
   reference ("E2 has no row 2015-13; its last row is 2016-09"). The v1 checker stays for
   saved runs, the corpus and the featured replays until they are re-recorded.
3. **Frontend, half a day to a day.** Each value is a reference with its label on hover and a
   click to its cell. "Try to fool the checker" changes meaning: a typed number in free text
   is always caught, and picking another cell rewrites the sentence so it stays true.
4. **Re-record and measure, Emil's tokens (about $3-6).** The six featured runs and the
   15-case eval on v2: first-attempt validity, rounds, steps, cost, readability. The bench
   becomes: swap any reference for any other cell and check the rendered sentence is true of
   it (by construction), plus a red team against the free-text lint.

Risk: the deadline. Freeze is Nov 3 and the video is Nov 4. If step 1 does not pass by
Oct 31 evening, keep v1 for the hackathon and say plainly in the README what it checks
(every number was found in the result it cites, and 28% of wrong-cell copies still pass).

## Stage 1, built (Oct 30)

`backend/src/wellpilot/agent/statements.py`, behind a flag: `run_agent(statements=True)`,
`python -m wellpilot.eval --statements`. Nothing changes for prose answers, the API or the
featured runs.

- **What the model sends:** submit_findings takes a summary and findings, each a list of
  parts: free words (`{"say": ...}`) or a statement (`value`, `change`, `compare`, `event`,
  `forecast`, `quote`) whose values are references into the tool results
  (`E1/wells[well=F-12]/water_cut`, `E7/monthly[month=2010-12]/water_cut`,
  `E5[kind=regime_change,signal=oil_rate]/details/before_mean`, `E13`, `E4/facts[2]`).
- **What code does:** resolves every reference (a bad one comes back with what is there:
  "no row of E7/monthly has month=2015-13 (its month values: 2008-02, ..., 2010-12)"),
  prints the sentence from the values' own labels (the screen's water cut is printed as a
  median over its 90 days, which the agent's prose never said), computes changes and
  ratios, refuses digits, dates, month names, units, amounts in words and quote marks in
  free words, refuses a change between two quantities or two wells, and keeps a fact off
  warned results, forecasts and causes. The printed answer becomes an ordinary Submission,
  so the answer panel, the second opinion and the reveal work as they are.
- **How it reads** (the 2010 time machine's results, written by hand as statements):
  "F-12's oil rate averaged 55% less after 2010-05-02 than before (4,207 Sm3/d before, 1,888
  Sm3/d after), while its choke opening fell from 83% to 58% over the 4 weeks either side of
  2010-05-02." "F-12 is forecast to make about 359,000 Sm3 of oil from 2011-01-01 to
  2011-12-31, range 324,000 to 470,000 Sm3 from the errors of 5 earlier forecasts made the
  same way." A wrong pick prints a true sentence about the row it took ("in 2010-06").
- **Not in stage 1:** the limit, share and job-effect types (compare and event cover them
  roughly), numbers that exist only in tool prose (the 2014 plug's 3,150 m), the UI showing
  references, replay of a statements run.
- **Tests:** `tests/test_statements.py` (printing on the frozen time machine's evidence,
  every refusal, and a scripted run sent back once and then verified).
- **The trial (Emil runs it, about $0.3):** `cd backend && uv run python -m wellpilot.eval
  --only time_machine_2010 --statements`. Go if it is verified in two submissions or fewer
  and the answer reads as well as the prose one; then stage 2. No go: keep v1 for the
  hackathon, as planned above.
