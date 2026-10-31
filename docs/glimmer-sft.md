# The checker as a teacher: fine-tuning Muse Glimmer

The checker labels every answer the agent submits. That is training data an open-weight
model can learn from: the answers it accepted, with the tool results they were written
from. The question is whether a model trained on them writes answers the checker accepts
more often than one that was only told the rules.

## What is measured

The task is the agent's last step: given the question and every tool result its run
gathered, write the `submit_findings` answer. No tool calling is needed, so Muse Glimmer,
a tuned Glimmer and Muse Spark Open answer exactly the same inputs.

- **Training** (`data/training/writer/train.jsonl`): 70 accepted answers, one per run, from
  13 questions; the agent's own system prompt and submit schema, the tool results, the
  answer. About 1.2M tokens, the longest example 35k (Glimmer reads 131k).
- **Test** (`test.jsonl`): 30 runs of 6 questions the training never sees: the dead gauge,
  the plan's water-cut limit, the 2014 restart, a refusal (F-99 does not exist), the
  end-2010 screen, and a well on the synthetic field.
- **Scored by:** the checker on the model's first answer (and on a second one after the
  checker's reply, as the loop gives it), the kinds of problem it finds, the eval's own
  regexes for that question (what a right answer must and must not say), style flags.
- **Baseline already in hand:** Muse Spark Open's first answers in those 25 live runs passed
  today's checker 5 times (20%).
- **Self-test, free:** the gold answers score 21 of 21 accepted and 41 of 41 regex checks;
  with a number made 17% larger, 19 of 21 are rejected (the other two had no number the
  helper could change).

The model is taught to pass the checker, so passing the checker is not the whole test: the
regexes and a read of a sample of answers are the check that it learned the field, not the
checker's blind spots.

## What it costs

- **Training:** Muse Glimmer 30B LoRA training is free on Fireworks' Serverless Training API
  until Nov 30. It runs from the local file, so nothing is uploaded as a dataset.
- **Answers:** Glimmer has no public inference endpoint, but a serverless training session
  samples its own checkpoints with no deployment. That is billed per token ($1.96 per million
  read, about 80% less when cached; $4.88 per million written). Base and tuned, both rounds on
  all 30 items, cost at most about $9 by `--dry-run`'s bound, and about $6-7 by the three-item
  check ($0.29 for items shorter than the average).
- **Muse Spark Open as a reference** (Baseten): about $1-2 with both rounds.

## Steps (Emil runs them; they use the Fireworks account)

`wellpilot.training_serverless` follows Fireworks' own example
(`training/examples/sft/muse_glimmer_starter.py` in `fw-ai/cookbook`) and needs its packages,
so it has its own venv. `data/` is git-ignored. The key is the `FIREWORKS_API_KEY` the backend
already reads.

**1. Install, once (about 5 minutes, about 3 GB).**

```bash
cd ~/wellpilot && git clone --depth 1 https://github.com/fw-ai/cookbook data/training/sft/cookbook
```

```bash
cd ~/wellpilot && uv venv data/training/sft/venv --python 3.13
```

```bash
cd ~/wellpilot && uv pip install --python data/training/sft/venv/bin/python -e data/training/sft/cookbook/training -e backend
```

**2. Check the data, free.** It renders every row with Glimmer's own tokenizer and sends
nothing. The default `--max-seq-len` of 40,960 fits every row (the longest has 35,222) and the
longest test prompt (39,131). Whether the shared pool takes rows this long shows in step 4.

```bash
cd ~/wellpilot/backend && ../data/training/sft/venv/bin/python -m wellpilot.training_serverless --dry-run
```

**3. Three questions on the base model, under $0.50.** This shows that the account has access
to serverless training and that Glimmer answers in JSON. If the reply says access must be
requested, ask Fireworks to enable serverless training for the account. If answers come back
cut off (`valid_json` below 3), add `--max-tokens 12000` to this and the next step.

```bash
cd ~/wellpilot/backend && ../data/training/sft/venv/bin/python -m wellpilot.training_serverless --epochs 0 --limit 3
```

**4. The run.** It trains first (3 epochs, 54 steps, a loss line per step), which is free, so a
row the pool refuses or a charge that should not be there shows before anything is spent. Then
it saves the adapter as the model `wellpilot-glimmer-writer` and scores base and tuned, about
20 minutes each at six at a time (base Glimmer took 1-6 minutes an item). How long a training
step takes on the shared pool is not known yet: the first lines will tell. `caffeinate` keeps
the Mac awake; keep the terminal open. A refusal for too many runs (`RESOURCE_EXHAUSTED`, 429)
means an earlier session still holds its slot: wait a few minutes and start again. After the
first steps, check the billing page shows no training charge; if it does, stop with Ctrl-C and
tell Claude.

```bash
cd ~/wellpilot/backend && caffeinate -is ../data/training/sft/venv/bin/python -m wellpilot.training_serverless
```

If it stops partway, every epoch's adapter is saved and its path is in
`data/training/writer/serverless/metrics.jsonl`. Score one with:

```bash
cd ~/wellpilot/backend && ../data/training/sft/venv/bin/python -m wellpilot.training_serverless --tuned-path <path>
```

**5. The reference on Muse Spark Open (Baseten, about $1-2 with both rounds).** It runs at the
agent's reasoning effort; left at Baseten's default, the reasoning used up `max_tokens` and most
replies came back empty (Oct 31). Three items first, to see answers come back:

```bash
cd ~/wellpilot/backend && uv run python -m wellpilot.training eval --label muse-check --limit 3
```

Then all of them:

```bash
cd ~/wellpilot/backend && uv run python -m wellpilot.training eval --label muse
```

Each scorecard is written to `data/evals/writer-<label>-<time>.json`. Tell Claude when the
three are there: it reads them and a sample of the answers and writes the result up, whatever
it is. If a command fails (a changed SDK call, an access error), paste its output.

The tuned model on a deployment (a machine at $8 an hour) is needed only to serve it through
the chat API, not for this test; `scripts/glimmer_eval.sh` does that when deployments work
again. The web form and `firectl dataset create` are the other way in, blocked since Oct 30 by
Fireworks' upload problem, and the same incident affects deployment creation.

## The result (Oct 31)

Training took 13 minutes (14 s a step) and the mean loss fell by epoch from 0.358 to 0.259 to
0.188, on 54 steps at rank 16. All 30 held-out items, up to two tries each:

| | Valid JSON | Accepted at once | Within two tries | Regexes | Tokens out per item | s per item |
|---|---|---|---|---|---|---|
| Muse Spark Open (medium reasoning) | 29 | 7 | 19 | 58 of 58 | 4,047 | 13 |
| Base Glimmer (high reasoning, the renderer's default) | 27 | 8 | 25 | 54 of 54 | 6,979 | 134 |
| Tuned on answers alone | 30 | 4 | 17 | 61 of 61 | 1,561 | 31 |
| Tuned with each answer's reasoning | 30 | 11 | 24 | 59 of 61 | 2,129 | 37 |

**First try, answers alone (70 examples):**

- **Training made it worse.** Within two tries, base beat tuned on 11 items and lost on 3
  (McNemar p = 0.057); at once, 7 against 3 (p = 0.34). Items of one question are not
  independent, so these p values flatter both. The losses are mostly the dead gauge (5 at once
  to 0) and the restart (6 within two tries to 3).
- **What it did learn:** the format. Every reply is valid JSON and a fifth as long.
- **Why (a rerun of six items, at most $0.17, verdicts the same as before):** the training
  answers hold no reasoning, and it answers straight away: its tokens are the reply alone, about
  770 an answer. Its slips are mechanical ("1,130,553" for 1,133,053, "from 100% to 30%"
  backwards, "185" for 185.7). It also took on Muse's length: 3.3 findings an answer against
  base's 2.8 and 1.57 problems a first try against 2.2, and 49 of the 70 examples are 5 questions.
- **Base against Muse:** within two tries base won 8 and lost 2 (p = 0.11). Three of base's
  misses were replies cut off at 8,000 tokens. This is the writing step alone, not the agent.
- **Cost:** sampling at most $2.79 and $2.16 before the prefill discount, plus $1.73 for Muse on
  Baseten. The Fireworks balance had not moved when this was written; billing may lag.

**Second try, the one variant named beforehand (Oct 31 night).** `WRITER_DIR=writer_reason`
`build --with-reasoning` keeps the model's reasoning from the step that submitted each answer (68
of 72 examples; the renderer trains on it as a `to=self` block before the answer, 117k trained
tokens against 82k). Loss by epoch 0.58, 0.46, 0.36; 54 steps; sampling at most $1.98.

- **The cause was right.** It beat the first tuned model on 9 items to 2 at once (p = 0.065) and
  10 to 3 within two tries (p = 0.092).
- **Level with base on acceptance:** 7 to 4 at once (p = 0.55), 4 to 5 within two tries (p = 1.0).
- **Better than base:** every reply valid JSON (30 against 27), a third of the tokens, a quarter
  of the time, 1.37 problems a first try against 1.57, 13 style flags against 17.
- **Its reasoning is short:** on answers accepted at once, about 1,370 tokens written for about
  800 of reply, so about 570 of reasoning.
- **Worse than base, and the checker missed it:** on P-1, whose planted cause is the stopped
  injector I-2, both answers fail the eval's regex; base, Muse and the first tuned model name
  I-2. One of the two was accepted at once and reads the field's falling voidage as support
  "remaining strong". Passing the checker is not the whole test, as this document warned.

Choices are not made on this test: 30 runs of 6 questions, read twice now. A third variant
(Glimmer's own accepted answers) would need a dev set first. No adapter is used in the product.
Scorecards are in `data/evals/`; tell them apart by the `label` inside (`glimmer-tuned` also
matches `glimmer-tuned-reason` as a file pattern).
