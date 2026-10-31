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
  all 30 items, cost at most about $9; `--dry-run` prints the bound for today's data.
- **Muse Spark Open as a reference** (Baseten): about 25 x 15k tokens in, about $0.5-0.8.

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

**3. Three questions on the base model, a few cents.** This shows that the account has access
to serverless training and that Glimmer answers in JSON. If the reply says access must be
requested, ask Fireworks to enable serverless training for the account. If answers come back
cut off (`valid_json` below 3), add `--max-tokens 12000` to this and the next step.

```bash
cd ~/wellpilot/backend && ../data/training/sft/venv/bin/python -m wellpilot.training_serverless --epochs 0 --limit 3
```

**4. The run.** It scores base Glimmer, trains (3 epochs, 54 steps, a loss line per step), saves
the adapter as the model `wellpilot-glimmer-writer` and scores the tuned one. How long a step
takes on the shared pool is not known yet: the first lines will tell. Keep the terminal open.
A refusal for too many runs (`RESOURCE_EXHAUSTED`, 429) means an earlier session still holds
its slot: wait a few minutes and start again.
After the first steps, check the billing page shows no training charge; if it does, stop with
Ctrl-C and tell Claude.

```bash
cd ~/wellpilot/backend && ../data/training/sft/venv/bin/python -m wellpilot.training_serverless
```

If it stops partway, every epoch's adapter is saved and its path is in
`data/training/writer/serverless/metrics.jsonl`. Score one with:

```bash
cd ~/wellpilot/backend && ../data/training/sft/venv/bin/python -m wellpilot.training_serverless --tuned-path <path>
```

**5. The reference on Muse Spark Open (Baseten, about $0.5-0.8).**

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

## What to expect, honestly

70 examples teach the format and the house rules (declare every number, hedge causes, no
facts on warned evidence, plain claims) faster than they teach the field. The likely gain is
in first-pass acceptance over base Glimmer; whether a tuned 30B model beats Muse Spark Open
on the regexes is open. If it does not help, the README says so.
