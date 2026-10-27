# Frozen demo runs

Saved runs replayed in CI by `tests/test_replay_runs.py`: the recorded model replies are
fed back, the tools and the checker run on today's code and the committed data fixture,
and the evidence results, check rounds and verdict must match the recording. A change that
would make a featured demo run behave differently on the live site fails the build.

Add a run after recording it on the frozen code (it costs tokens; Emil runs it):

    cp data/runs/<id>.json backend/tests/fixtures/runs/
    cd backend && uv run python -m wellpilot.agent.replay ../data/runs/<id>.json   # must match

The same ids go into FEATURED_RUNS for the public deployment.

Run fcfb4188de3b was taken out on Oct 27: its accepted summary states "about 7.89 times
faster", a ratio calculated from the r2_log 0.019 plateau fit, as settled. The checker now
sends such a summary back (summary numbers from evidence with warnings must be hedged), so
the recording no longer replays as recorded. Re-record the demo runs and add them here.
