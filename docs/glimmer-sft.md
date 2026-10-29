# The checker as a teacher: fine-tuning Muse Glimmer

The checker labels every answer the agent submits. That is training data an open-weight
model can learn from: the answers it accepted, with the tool results they were written
from. The question is whether a model trained on them writes answers the checker accepts
more often than one that was only told the rules.

## What is measured

The task is the agent's last step: given the question and every tool result its run
gathered, write the `submit_findings` answer. No tool calling is needed, so Muse Glimmer,
a tuned Glimmer and Muse Spark Open answer exactly the same inputs.

- **Training** (`data/training/writer/train.jsonl`): 73 accepted answers, one per run, from
  13 questions; the agent's own system prompt and submit schema, the tool results, the
  answer. About 1M tokens, the longest example about 25k (Glimmer reads 131k).
- **Test** (`test.jsonl`): 25 runs of 6 questions the training never sees: the dead gauge,
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

- **Training:** Muse Glimmer 30B supervised fine-tuning is free on Fireworks until Nov 30
  (LoRA, via the web UI or the Serverless Training API).
- **Answers:** Glimmer has no serverless endpoint on Fireworks, and a LoRA runs only on an
  on-demand deployment: an H100 at $8 an hour, billed by the second, start-up not charged.
  One deployment of the base model with add-ons serves both base and tuned Glimmer; the 25
  items twice each take well under an hour. Delete the deployment when done.
- **Muse Spark Open as a reference** (Baseten): about 25 x 15k tokens in, about $0.5-0.8.

## Steps (Emil runs them; they use the Fireworks account)

1. Redeem the Fireworks promo code before Nov 3 (Billing -> Credits -> Redeem Promo).
2. Build the data (free; already built, rerun after the corpus grows):

   ```bash
   cd backend && uv run python -m wellpilot.training build
   ```

3. In the Fireworks console: Datasets -> upload `data/training/writer/train.jsonl`; then
   Fine-tuning -> new supervised job: base model `accounts/fireworks/models/muse-glimmer-30b`,
   the dataset, LoRA defaults (rank 16), 2 epochs, max context at least 32768, output model
   `wellpilot-glimmer-writer`.
4. When it finishes, deploy the base model with add-ons and load the adapter (with `firectl`,
   or Deployments in the console):

   ```bash
   firectl deployment create accounts/fireworks/models/muse-glimmer-30b --deployment-shape <bf16 shape> --enable-addons
   ```

   ```bash
   firectl load-lora wellpilot-glimmer-writer --deployment <deployment id>
   ```

5. Score base, tuned, and Muse Spark Open (from `backend/`):

   ```bash
   LLM_PROVIDER=fireworks FIREWORKS_MODEL="accounts/fireworks/models/muse-glimmer-30b#<deployment id>" uv run python -m wellpilot.training eval --label glimmer-base
   ```

   ```bash
   LLM_PROVIDER=fireworks FIREWORKS_MODEL="accounts/<account>/models/wellpilot-glimmer-writer#<deployment id>" uv run python -m wellpilot.training eval --label glimmer-tuned
   ```

   ```bash
   uv run python -m wellpilot.training eval --label muse
   ```

6. Delete the deployment:

   ```bash
   firectl deployment delete <deployment id>
   ```

Each scorecard goes to `data/evals/writer-<label>-<time>.json`. Claude then reads them and
a sample of the answers and writes the result up, whatever it is.

## What to expect, honestly

73 examples teach the format and the house rules (declare every number, hedge causes, no
facts on warned evidence, plain claims) faster than they teach the field. The likely gain is
in first-pass acceptance over base Glimmer; whether a tuned 30B model beats Muse Spark Open
on the regexes is open. If it does not help, the README says so.
