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

- 8aef8d548380, the time machine: "Which wells need attention right now, and why? Forecast
  the next 12 months of oil for the well that needs it most." with data up to 2010-12-31.
  F-12 first (dead gauge; rate 55% lower since May 2010, a gradual shift); its 2011 forecast
  359k Sm3 against 340k produced (+6%), the whole-history fit 685k (+102%), the band from
  past errors held. Verified on attempt 2; 9 steps, about $0.23.
- 3fc72959335a, the plan's forecast chart read by the model (look_at_page): the text says
  9 000 Sm3/d for about 19 months, the figure as drawn about 8 400 Sm3/d for about 2 years.
  Verified on attempt 2; 5 steps, about $0.07.

Earlier frozen runs (fcfb4188de3b, c8b6cbb62280, then cc62fd10242e and 44f49414e399 when the
regime-change detector changed) were retired when the checker and the tools changed what
they would return.

e51eada95269 (F-12's decline against the 2005 plan) was retired on Oct 28 night: its
finding 5, "The field stayed within plan design limits", is now rejected, since its own E10
has the field's water cut at or above the plan's 0.80 for 27 months (the third judge
panel's catch). It stays in `tests/fixtures/retired_runs/` for the red-team tests until the
question is re-recorded.
