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

About 30 minutes of hands-on time, spread over an evening; training itself runs on its own.

**1. Prepare (once, about 5 minutes).**
- Redeem the Fireworks promo code (Billing -> Credits -> Redeem Promo); it must be done by Nov 3.
- Install the CLI and sign in; `whoami` prints the account id the commands below need:

  ```bash
  brew tap fw-ai/firectl
  ```

  ```bash
  brew install firectl
  ```

  ```bash
  firectl signin
  ```

  ```bash
  firectl whoami
  ```

**2. Upload the data and start training (about 5 minutes; then it runs by itself).**

```bash
cd ~/wellpilot && firectl dataset create wellpilot-writer data/training/writer/train.jsonl
```

```bash
firectl sftj create --base-model accounts/fireworks/models/muse-glimmer-30b --dataset wellpilot-writer --output-model wellpilot-glimmer-writer --epochs 2 --lora-rank 16 --max-context-length 32768
```

Check on it with `firectl sftj get <job id>` (the id is printed by the command above). If it
says examples were cut at the context limit, start it again with `--max-context-length 65536`.
Training is free for Glimmer until Nov 30.

If `firectl dataset create` says dataset uploads are disabled for the account (it did on
Oct 30), use the console's training form instead (Fine-tuning -> new job): model Muse Glimmer
30B, method SFT, upload `train.jsonl` as the dataset; for evaluation upload
`data/training/writer/validation.jsonl` (the held-out questions' accepted answers) rather
than splitting 20% off the 73 training examples; epochs 2, LoRA rank 16, max context 32768
(the others as they are). The estimate must show the training as free; if it shows a cost,
stop and check.

**3. While it trains: the reference on Muse Spark Open (Baseten, about $0.5-0.8).**

```bash
cd ~/wellpilot/backend && uv run python -m wellpilot.training eval --label muse
```

**4. When training is done: one machine for both Glimmers (billing starts here, $8 an hour).**
- Find a deployment shape for the base model; pick a BF16 one (FP8 and FP4 shapes cannot
  hold an adapter). If the list is unclear, send it to Claude.

  ```bash
  firectl deployment-shape-version match --model accounts/fireworks/models/muse-glimmer-30b
  ```

- Create the deployment with add-ons, wait until `firectl deployment get <deployment id>`
  says it is ready, then load the trained adapter:

  ```bash
  firectl deployment create accounts/fireworks/models/muse-glimmer-30b --deployment-shape <BF16 shape> --enable-addons
  ```

  ```bash
  firectl load-lora wellpilot-glimmer-writer --deployment <deployment id>
  ```

**5. Score both (about 20-40 minutes in all).** First try three items, to see the model
answers at all; then the full set. Replace `<account>` and `<deployment id>`:

```bash
cd ~/wellpilot/backend && LLM_PROVIDER=fireworks FIREWORKS_MODEL="accounts/fireworks/models/muse-glimmer-30b#accounts/<account>/deployments/<deployment id>" uv run python -m wellpilot.training eval --label glimmer-base --limit 3
```

```bash
cd ~/wellpilot/backend && LLM_PROVIDER=fireworks FIREWORKS_MODEL="accounts/fireworks/models/muse-glimmer-30b#accounts/<account>/deployments/<deployment id>" uv run python -m wellpilot.training eval --label glimmer-base
```

```bash
cd ~/wellpilot/backend && LLM_PROVIDER=fireworks FIREWORKS_MODEL="accounts/<account>/models/wellpilot-glimmer-writer#accounts/<account>/deployments/<deployment id>" uv run python -m wellpilot.training eval --label glimmer-tuned
```

**6. Stop the billing.**

```bash
firectl deployment delete <deployment id>
```

```bash
firectl deployment list
```

Each scorecard is written to `data/evals/writer-<label>-<time>.json`. Tell Claude when the
three are there: it reads them and a sample of the answers and writes the result up,
whatever it is. If a command fails (a model name format, a shape), paste its output.

## What to expect, honestly

73 examples teach the format and the house rules (declare every number, hedge causes, no
facts on warned evidence, plain claims) faster than they teach the field. The likely gain is
in first-pass acceptance over base Glimmer; whether a tuned 30B model beats Muse Spark Open
on the regexes is open. If it does not help, the README says so.
