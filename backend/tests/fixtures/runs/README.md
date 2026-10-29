# Frozen demo runs

Saved runs replayed in CI by `tests/test_replay_runs.py`: the recorded model replies are
fed back, the tools and the checker run on today's code and the committed data fixture,
and the evidence results, check rounds and verdict must match the recording. A change that
would make a featured demo run behave differently on the live site fails the build.

Add a run after recording it on the frozen code (it costs tokens; Emil runs it):

    cp data/runs/<id>.json backend/tests/fixtures/runs/
    cd backend && uv run python -m wellpilot.agent.replay ../data/runs/<id>.json   # must match

The same ids go into FEATURED_RUNS for the public deployment.

The featured runs (Muse Spark Open on Baseten, recorded by the eval on Oct 28-29;
FEATURED_RUNS lists them in this order):

- 69519dd0a995, the time machine: "Which wells need attention right now, and why? Forecast
  the next 12 months of oil for the well that needs it most." with data up to 2010-12-31.
  F-12 first (dead gauge since 2010-10-12; rate 55% lower on average after 2010-05-02,
  when the choke was cut from 83% to 58%). Its forecast fits from the shift it found:
  about 359k Sm3 for 2011, range 324k-470k, against 340k produced. Verified on attempt 3,
  second opinion sound. 9 steps, 19 tool calls, 44 s, $0.31.
- 73a1757ab912, what to do with F-12, as of 2014-10-31: the 2010 plug at 3,210 m already
  shuts off what enters below it, so the gaps there cut nothing new; the log is 1,500 days
  old, so a new log before any job; on that log the only gap left, 3,147-3,161 m, would cut
  about 98% of the water and 81% of the oil still flowing. The operator's plug three weeks
  later was in that gap, at 3,150 m, with a new zone above. Verified on attempt 3, second
  opinion sound. 10 steps, 14 tool calls, 80 s, $0.39. Replaces dca494b07eb4, which picked
  the gap the 2010 plug already sat in.
- 6af55f1b3eef, F-12's decline against the 2005 plan: oil averaged 72% less after
  2010-07-18 than before, with only a small change across the date; water cut shifted
  gradually from 3% to 51% around 2010-02-14; liquid held near 5,130 Sm3/d while the choke
  opened to 100% by 2012; breakthrough at about 21 months against the plan's "about 2 years",
  and the plan's 6 MSm3 at that point. Verified on attempt 3, second opinion sound. 10 steps,
  18 tool calls, $0.46. Replaces c6b4a6b047a8 ("not a choke cut", "matched the 6 million").
- c8501c918af6, the plan's forecast chart: the text's 9,000 Sm3/d for about 19 months, the
  figure's plateau at about 8,460 Sm3/d for 19 months (2008-03 to 2009-09). Recorded on
  Oct 28 before SAM 3.1's values were hidden from the model: the answer gives SAM's
  measurement, where the model's own reading was 8,400 Sm3/d. 5 steps, $0.08.
- 83a3114d185f, F-12's 2014 water shut-off: the 7-inch EVO plug at 3150 m set 2014-11-26
  and the new perforation 3126.2-3130.7 m (water cut 0.97 to 0.34, oil 177 to 857 Sm3/d in
  the 60 days after), against the 2010 plug at 3210 m in the blank pipe below most of the
  flow. Verified on attempt 3, second opinion sound. 6 steps, 10 tool calls, 37 s, $0.24.
  Replaces 485c17df2acb ("shut in on 2014-11-30 for a workover").
- f7d3faebb139, the agent's own Python: F-12's monthly water cut against its choke over
  2011-2013 in the sandbox; all findings on it are inferences. 7 steps, 28 s, $0.18.
- 3fc72959335a, an earlier reading of the plan's chart, kept for its replay (not featured).

On Oct 29 detect_events began giving a water-cut change's level across its date, and
waterflood's note began saying its reference water cuts are not an economic limit: the
evidence of 69519dd0a995 and f7d3faebb139 differs there (listed in `DRIFT` in the test),
their check rounds and verdicts replay exactly.

Retired (in `../retired_runs/`, still read by tests): e51eada95269 (its finding 5 is now
rejected) and 0d95b4d9cf46 (Oct 28: its "maximum water cut 67% below the 80% limit" reads
as a difference from the limit, and "oil about half its earlier rate" writes a number in
words; both are now sent back, so it no longer replays as recorded). Re-record the time
machine to replace it. a5dc0b339ff9 (the agent's own Python,
Oct 28) is retired too: its finding 1 is a fact resting on run_python, and what code
returns is now always an inference. Re-record the own_code_f12 eval case to replace it. Also retired on Oct 28 evening, after the water-spike cleaning changed
their evidence: 8aef8d548380 (the earlier time machine), b2b80c7484ee (decline vs plan,
"collapsed"), b4396314f788 (the earlier water shut-off) and 033bb064359f (the own-code run
that typed its data into the code). On Oct 29: 4f938683446e, 85db026c4c7f and
bf4fa9e48aa3 (production_log now dates a plug by the report that names it and measures it
60 days either side; 4f93's final answer is now rejected). Later on Oct 29, after the eighth
panel: e10e0aaf0542 ("prepare for shut-in", "no economic oil"), e08be5885cb4 ("water cut
jumped" for period means, now sent back), 063034849654 (Muse's disputed "2 years" given as the
figure's, now sent back) and 3d27b4a64831, replaced by the runs above.
