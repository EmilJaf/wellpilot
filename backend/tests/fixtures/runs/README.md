# Frozen demo runs

Saved runs replayed in CI by `tests/test_replay_runs.py`: the recorded model replies are
fed back, the tools and the checker run on today's code and the committed data fixture,
and the evidence results, check rounds and verdict must match the recording. A change that
would make a featured demo run behave differently on the live site fails the build.

Add a run after recording it on the frozen code (it costs tokens; Emil runs it):

    cp data/runs/<id>.json backend/tests/fixtures/runs/
    cd backend && uv run python -m wellpilot.agent.replay ../data/runs/<id>.json   # must match

The same ids go into FEATURED_RUNS for the public deployment.

The featured runs (Oct 27 evening, Muse Spark Open on Baseten, recorded by the eval):

- cc62fd10242e, the time machine: "Which wells need attention right now, and why? Forecast
  the next 12 months of oil for the well that needs it most." with data up to 2010-12-31.
  F-12 first (dead gauge, 55% step down); its 2011 forecast 359k Sm3 against 340k produced
  (+6%), the whole-history fit 685k (+102%). Verified on attempt 2; 7 steps, about $0.15.
- 44f49414e399, F-12's decline against the 2005 plan: breakthrough after about 2 years as
  planned, liquid held while the choke opened, the field at its design water cut. Verified
  on attempt 3; 11 steps, about $0.48.
- 3fc72959335a, the plan's forecast chart read by the model (look_at_page): the text says
  9 000 Sm3/d for about 19 months, the figure as drawn about 8 400 Sm3/d for about 2 years.
  Verified on attempt 2; 5 steps, about $0.07.

Earlier frozen runs (fcfb4188de3b, c8b6cbb62280) were retired when the checker and the
tools changed what they would return.
