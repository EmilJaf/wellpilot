# Frozen demo runs

Saved runs replayed in CI by `tests/test_replay_runs.py`: the recorded model replies are
fed back, the tools and the checker run on today's code and the committed data fixture,
and the evidence results, check rounds and verdict must match the recording. A change that
would make a featured demo run behave differently on the live site fails the build.

Add a run after recording it on the frozen code (it costs tokens; Emil runs it):

    cp data/runs/<id>.json backend/tests/fixtures/runs/
    cd backend && uv run python -m wellpilot.agent.replay ../data/runs/<id>.json   # must match

The same ids go into FEATURED_RUNS for the public deployment.

The featured runs (Muse Spark Open on Baseten, recorded by the eval on Oct 28 evening, after
the fifth judge panel's fixes and the water-spike cleaning):

- 9291bac14418, the time machine: "Which wells need attention right now, and why? Forecast
  the next 12 months of oil for the well that needs it most." with data up to 2010-12-31.
  F-12 first (dead gauge, a monitoring gap; rate 55% lower since May 2010). Its cited
  forecast fits from the 2010-05-02 shift it found: 359k Sm3 for 2011 against 340k produced
  (+6%: the rate 12% low, the time on stream 20% high); its range from 5 past errors held.
  The checker's catch on round 1 was one of meaning: "69%" named water where E1 has the
  water cut. Second opinion sound. 8 steps, 17 tool calls, 36 s, $0.23.
- 5ceeb3c745cc, F-12's decline against the 2005 plan: breakthrough timing against "about 2
  years", the 2010-07-18 change as a gradual shift, the decline after it, the production
  logs on the 2010 plug, the field reaching 0.80 in June 2012. Verified on attempt 3, second
  opinion sound. 8 steps, $0.37.
- c8501c918af6, the plan's forecast chart (Oct 28): Muse Spark Open read the plateau at
  about 8 400 Sm3/d, SAM 3.1's measurement of the same curve is 8 464 (within the 250 Sm3/d
  a quarter gridline allows). Its length reading (a year, 2008, for "2 to 2.5 years") was
  not compared then; today's cross-check reads the length from the model's words. 5 steps,
  $0.08.
- 3dbb4c8e390f, F-12's 2014 water shut-off: the plug at 3150 m and the new zone above it,
  against the 2010 plug at 3210 m, from the daily drilling reports and the production logs
  (the 2009 one read off its pictures by SAM 3.1). Verified on attempt 2, second opinion
  sound. 9 steps, 46 s, $0.33.
- b94856e6bedd, the agent's own Python: F-12's water cut against its choke over 2011-2013,
  read from the daily data in the sandbox (971 rows): Pearson 0.40 on monthly means, 0.41
  daily; all findings on it are inferences. 5 steps, $0.12.
- 3fc72959335a, the plan's chart read by the model, kept for its replay (not featured).

Retired (in `../retired_runs/`, still read by tests): e51eada95269 (its finding 5 is now
rejected) and 0d95b4d9cf46 (Oct 28: its "maximum water cut 67% below the 80% limit" reads
as a difference from the limit, and "oil about half its earlier rate" writes a number in
words; both are now sent back, so it no longer replays as recorded). Re-record the time
machine to replace it. a5dc0b339ff9 (the agent's own Python,
Oct 28) is retired too: its finding 1 is a fact resting on run_python, and what code
returns is now always an inference. Re-record the own_code_f12 eval case to replace it. Also retired on Oct 28 evening, after the water-spike cleaning changed
their evidence: 8aef8d548380 (the earlier time machine), b2b80c7484ee (decline vs plan,
"collapsed"), b4396314f788 (the earlier water shut-off) and 033bb064359f (the own-code run
that typed its data into the code).
