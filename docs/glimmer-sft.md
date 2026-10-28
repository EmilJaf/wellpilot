# The checker as a teacher: fine-tuning Muse Glimmer

The checker labels every answer the agent submits. `python -m wellpilot.agent.corpus export`
turns the reviewed corpus into training data in `data/training/`:

- `sft.jsonl`: 84 accepted answers as chat examples (`{"messages": [system, user, assistant]}`):
  the question and the tool results in, the checked submission out.
- `preferences.jsonl`: 79 pairs, an answer the checker sent back and the one it accepted next
  in the same run, with what it caught.
- `accepted.jsonl`: the accepted answers with the evidence they cite.

Labels are the reviewed verdicts, so a false rejection corrected in review is not taught as
wrong. 19 questions only: enough to show the loop, not to claim a better model.

## Running it (needs the Fireworks key; supervised fine-tuning of Muse Glimmer is free there
until Nov 30)

1. Check the base model's id in the Fireworks model catalog (Muse Glimmer 30B) and that its
   fine-tuning accepts chat examples of this length (the longest example is about 60k
   characters of tool results).
2. Upload the data and start the job with `firectl` (or the web console's fine-tuning page):

   ```bash
   firectl create dataset wellpilot-sft data/training/sft.jsonl
   ```

   ```bash
   firectl create sftj --base-model <glimmer model id> --dataset wellpilot-sft --output-model wellpilot-glimmer
   ```

3. Evaluate before claiming anything: hold out whole questions (not answers to a question
   already seen), run the tuned model through `python -m wellpilot.eval` with
   `LLM_PROVIDER=fireworks` and the tuned model's id, and compare the share verified on the
   first attempt with the base model's on the same questions.
