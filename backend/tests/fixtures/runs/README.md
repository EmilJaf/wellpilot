# Frozen demo runs

Saved runs replayed in CI by `tests/test_replay_runs.py`: the recorded model replies are
fed back, the tools and the checker run on today's code and the committed data fixture,
and the evidence results, check rounds and verdict must match the recording. A change that
would make a featured demo run behave differently on the live site fails the build.

Add a run after recording it on the frozen code (it costs tokens; Emil runs it):

    cp data/runs/<id>.json backend/tests/fixtures/runs/
    cd backend && uv run python -m wellpilot.agent.replay ../data/runs/<id>.json   # must match

The same ids go into FEATURED_RUNS for the public deployment.

The featured runs (Muse Spark Open on Baseten, recorded by the eval on Oct 28-29, after the
seventh judge panel's fixes; FEATURED_RUNS lists them in this order):

- 69519dd0a995, the time machine: "Which wells need attention right now, and why? Forecast
  the next 12 months of oil for the well that needs it most." with data up to 2010-12-31.
  F-12 first (dead gauge since 2010-10-12; rate shifted down 55% around 2010-05-02 while the
  choke was cut from 83% to 58%). Its forecast fits from the shift it found: about 359k Sm3
  for 2011, range 324k-470k, against 340k produced. Verified on attempt 3, second opinion
  sound. 9 steps, 19 tool calls, 44 s, $0.31.
- e10e0aaf0542, what to do with F-12, as of 2014-10-31: the 2010 plug at 3,210 m did not
  help; a plug at about 3,185 m in blank pipe would cut about 31% of the water on the 2010
  log; the log is old, so a new log first, else prepare for shut-in (about 42,800 Sm3 left
  in 12 months). The operator's job three weeks later (plug at 3,150 m, a new zone above
  it) paid: the reveal shows the agent was too cautious. Verified on attempt 2, second
  opinion sound. 8 steps, 13 tool calls, 54 s, $0.32.
- e08be5885cb4, F-12's decline against the 2005 plan: oil averaged 72% less after
  2010-07-18 than before (a gradual shift), about 0.45 a year after it, breakthrough after
  about 21 months against the plan's "about 2 years", the field at the 80% design water cut
  from June 2012 and never over its liquid or water handling limits. Verified on attempt 3
  (round 1 holds one false rejection, since fixed: "86%" tied to June 2012 across "and").
  7 steps, 15 tool calls, 35 s, $0.28.
- 063034849654, the plan's forecast chart: the text's 9,000 Sm3/d for about 19 months, the
  figure read at about 8,400 Sm3/d for about 2 years (SAM 3.1 agrees on the rate, not the
  length). Verified on attempt 2. 4 steps, 13 s, $0.06.
- 3d27b4a64831, F-12's 2014 water shut-off: the plug at 3,150 m and the new zone above it,
  against the 2010 plug at 3,210 m (set 2010-09-23, below only 33-38% of the water on the
  2010 log), from the daily drilling reports and the production logs. Verified on attempt 2,
  second opinion sound. 8 steps, 12 tool calls, 51 s, $0.28.
- f7d3faebb139, the agent's own Python: F-12's monthly water cut against its choke over
  2011-2013 in the sandbox; all findings on it are inferences. 7 steps, 28 s, $0.18.
- 3fc72959335a and c8501c918af6, earlier readings of the plan's chart, kept for their
  replays (not featured).

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
60 days either side; 4f93's final answer is now rejected), replaced by the runs above.
