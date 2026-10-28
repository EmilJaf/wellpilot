# Frozen demo runs

Saved runs replayed in CI by `tests/test_replay_runs.py`: the recorded model replies are
fed back, the tools and the checker run on today's code and the committed data fixture,
and the evidence results, check rounds and verdict must match the recording. A change that
would make a featured demo run behave differently on the live site fails the build.

Add a run after recording it on the frozen code (it costs tokens; Emil runs it):

    cp data/runs/<id>.json backend/tests/fixtures/runs/
    cd backend && uv run python -m wellpilot.agent.replay ../data/runs/<id>.json   # must match

The same ids go into FEATURED_RUNS for the public deployment.

The featured runs (Muse Spark Open on Baseten, recorded by the eval):

- 033bb064359f, the agent's own Python (Oct 28 afternoon, replaces a5dc0b339ff9): F-12's
  monthly water cut against its choke over 2011-2013. run_python gave Pearson 0.68 and
  Spearman 0.48 over 36 months; every finding resting on it is an inference, and the
  checker sent the summary back twice until it hedged the code's numbers. Second opinion
  sound. 6 steps, $0.15.
- 8aef8d548380, the time machine as served (again from Oct 28, after an outside review): "Which wells need attention right now, and why? Forecast
  the next 12 months of oil for the well that needs it most." with data up to 2010-12-31.
  F-12 first (dead gauge; rate 55% lower since May 2010, a gradual shift); its 2011 forecast
  359k Sm3 against 340k produced (+6%), the whole-history fit 685k (+102%), the band from
  past errors held. Verified on attempt 2; 9 steps, about $0.23.
- 3fc72959335a, the plan's forecast chart read by the model (look_at_page): the text says
  9 000 Sm3/d for about 19 months, the figure as drawn about 8 400 Sm3/d for about 2 years.
  Verified on attempt 2; 5 steps, about $0.07.
- b2b80c7484ee, F-12's decline against the 2005 plan (Oct 28): water broke through at 21
  months against the plan's "about 2 years"; the 2010-07-18 change; the production logs
  show the 2010 plug at 3210 m below about 1% of the oil and 33% of the water. The second
  opinion raised three points and the agent revised its answer, verified again. 10 steps,
  about $0.53 ($0.31 to the first checked answer).
- c8501c918af6, the plan's forecast chart (Oct 28): Muse Spark Open read the plateau at
  about 8 400 Sm3/d, SAM 3.1's measurement of the same curve is 8 464 (within the 250 Sm3/d
  a quarter gridline allows): the two Meta models agree on one picture. 5 steps, $0.08.
- b4396314f788, F-12's 2014 water shut-off (Oct 28): the plug at 3150 m and the new zone
  above it, against the 2010 plug at 3210 m, from the daily drilling reports and the
  production logs (the 2009 one read off its pictures by SAM 3.1). 6 steps, $0.19.

Earlier frozen runs (fcfb4188de3b, c8b6cbb62280, then cc62fd10242e and 44f49414e399 when the
regime-change detector changed) were retired when the checker and the tools changed what
they would return.

e51eada95269 (F-12's decline against the 2005 plan) was retired on Oct 28 night: its
finding 5, "The field stayed within plan design limits", is now rejected, since its own E10
has the field's water cut at or above the plan's 0.80 for 27 months (the third judge
panel's catch). It stays in `tests/fixtures/retired_runs/` for the red-team tests until the
question is re-recorded.

Retired (in `../retired_runs/`, still read by tests): e51eada95269 (its finding 5 is now
rejected) and 0d95b4d9cf46 (Oct 28: its "maximum water cut 67% below the 80% limit" reads
as a difference from the limit, and "oil about half its earlier rate" writes a number in
words; both are now sent back, so it no longer replays as recorded). Re-record the time
machine to replace it. a5dc0b339ff9 (the agent's own Python,
Oct 28) is retired too: its finding 1 is a fact resting on run_python, and what code
returns is now always an inference. Re-record the own_code_f12 eval case to replace it.
