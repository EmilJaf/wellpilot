"""Train Muse Glimmer on the accepted answers through Fireworks' Serverless Training API and
score base and tuned on the held-out questions in the same session: no dataset to upload, no
deployment to pay for by the hour. Training is free until Nov 30; sampling is billed per token.

The calls follow fw-ai/cookbook, training/examples/sft/muse_glimmer_starter.py. They need that
cookbook's packages, so this runs in its own venv, not the backend's; docs/glimmer-sft.md has
the steps.

    python -m wellpilot.training_serverless --dry-run              # free: renders every row
    python -m wellpilot.training_serverless --epochs 0 --limit 3   # cents: does it answer at all
    python -m wellpilot.training_serverless                        # base, training, tuned
    python -m wellpilot.training_serverless --tuned-path <path>    # score a saved checkpoint
"""

import argparse
import asyncio
import json
import random
import sys

from wellpilot.config import get_settings
from wellpilot.training import _out, evaluate, save_card, system_prompt, user_message

BASE_MODEL = "accounts/fireworks/models/muse-glimmer-30b"
TOKENIZER_MODEL = "meta-models/Muse-Glimmer-30B"
RENDERER_NAME = "muse_glimmer"
SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"
GATEWAY_URL = "https://api.fireworks.ai"
OUTPUT_MODEL = "wellpilot-glimmer-writer"
LORA_ALPHA = 32
LORA_TARGETS = {"train_mlp": True, "train_attn": True, "train_unembed": True}
# USD per token on the docs' serverless table, Oct 31. Prefill is about 80% less when cached.
PRICE_PREFILL, PRICE_SAMPLE = 1.96e-6, 4.88e-6


def rows(name: str) -> list[dict]:
    path = _out() / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def render(examples: list[dict], renderer, train_on, max_seq_len: int) -> list[tuple[list, list]]:
    """Each example's tokens and the weights that mark its answer. A row too long stops the
    run here, before anything is sent: cutting one would train on half an answer."""
    rendered = []
    for n, example in enumerate(examples):
        model_input, weights = renderer.build_supervised_example(
            example["messages"], train_on_what=train_on
        )
        tokens = [int(t) for t in model_input.to_ints()]
        marked = [float(w) for w in weights.tolist()]
        if len(tokens) > max_seq_len:
            sys.exit(f"row {n} has {len(tokens)} tokens, over --max-seq-len {max_seq_len}")
        if not any(w > 0 for w in marked[1:]):
            sys.exit(f"row {n} has no answer tokens to train on")
        rendered.append((tokens, marked))
    return rendered


def datums(rendered: list[tuple[list, list]], sdk) -> list:
    return [
        sdk.Datum(
            model_input=sdk.ModelInput.from_ints(tokens[:-1]),
            loss_fn_inputs={"target_tokens": tokens[1:], "weights": marked[1:]},
        )
        for tokens, marked in rendered
    ]


def train(client, data: list, sdk, args, log, checkpoint) -> int:
    """Adam over shuffled batches, a saved adapter after each epoch. The steps taken."""
    adam = sdk.AdamParams(
        learning_rate=args.lr,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        grad_clip_norm=1.0,
    )
    rng = random.Random(args.seed)
    step = 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = [data[i] for i in order[start : start + args.batch_size]]
            done = client.forward_backward(batch, "cross_entropy").result()
            opt = client.optim_step(adam, emit_grad_norm_metrics=True).result()
            step += 1
            metrics = dict(getattr(done, "metrics", None) or {})
            answered = metrics.get("response_tokens") or metrics.get("num_loss_tokens") or 1.0
            grads = {k: v for k, v in (getattr(opt, "metrics", None) or {}).items() if "grad" in k}
            loss = metrics.get("loss:sum", 0.0) / max(float(answered), 1.0)
            log({"step": step, "epoch": epoch, "loss": round(loss, 4), **grads})
        checkpoint(f"wp-epoch-{epoch}")  # names: 17 characters at most, one use each
    return step


def sampler_ask(sampler, renderer, sdk, get_text):
    """evaluate()'s `ask`, on a sampler: the same prompt a chat model would get, tokenized."""

    async def ask(messages: list[dict], max_tokens: int) -> tuple[str, int, int]:
        prompt = renderer.build_generation_prompt(messages)
        params = sdk.SamplingParams(
            max_tokens=max_tokens, temperature=0.0, stop=renderer.get_stop_sequences()
        )
        future = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)
        result = await asyncio.to_thread(future.result) if hasattr(future, "result") else future
        tokens = list(result.sequences[0].tokens)
        message, _ = renderer.parse_response(tokens)
        return get_text(message), len(prompt.to_ints()), len(tokens)

    return ask


def costs(results: list[dict]) -> float:
    """Sampling before any prefill discount, so an upper bound."""
    return (
        sum(r["tokens_in"] for r in results) * PRICE_PREFILL
        + sum(r["tokens_out"] for r in results) * PRICE_SAMPLE
    )


def dry_run(train_rows, items: list[dict], rendered: list, renderer, tokenizer, args) -> None:
    """What the run would send, with no key and no call to Fireworks."""
    lengths = [len(tokens) for tokens, _ in rendered]
    answers = [sum(1 for w in marked if w > 0) for _, marked in rendered]
    prompts = [
        len(
            renderer.build_generation_prompt(
                [
                    {"role": "system", "content": system_prompt(item["as_of"])},
                    {"role": "user", "content": user_message(item["question"], item["evidence"])},
                ]
            ).to_ints()
        )
        for item in items
    ]
    steps = args.epochs * -(-len(rendered) // args.batch_size)
    # Two models, each up to two rounds; a second round reads the first answer as well.
    tokens_in = 2 * args.rounds * sum(prompts)
    tokens_out = 2 * args.rounds * len(items) * args.max_tokens
    print(
        json.dumps(
            {
                "train_rows": len(train_rows),
                "train_tokens": sum(lengths),
                "longest_row": max(lengths),
                "answer_tokens": sum(answers),
                "steps": steps,
                "test_items": len(items),
                "longest_test_prompt": max(prompts),
                "sampling_usd_at_most": round(
                    tokens_in * PRICE_PREFILL + tokens_out * PRICE_SAMPLE, 2
                ),
            },
            indent=1,
        )
    )
    text = tokenizer.decode(rendered[0][0])
    print(f"first row starts: {text[:400]!r}\nand ends: {text[-400:]!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="render everything, send nothing")
    parser.add_argument("--epochs", type=int, default=3, help="0 scores the base model only")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N test items")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-base", action="store_true", help="leave out the base model's score")
    parser.add_argument("--no-promote", action="store_true", help="do not save a LoRA model")
    parser.add_argument("--tuned-path", help="score this saved checkpoint and do nothing else")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Imported here: they exist only in the cookbook's venv.
    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import TrainOnWhat, get_renderer, get_text_content
    from training.renderer.tokenizer import get_tokenizer
    from training.utils.serverless import enable_serverless_supervised_409_retry

    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    renderer = get_renderer(RENDERER_NAME, tokenizer)
    train_rows = rows("train")
    items = rows("test")
    if args.limit:
        items = items[: args.limit]
    rendered = render(train_rows, renderer, TrainOnWhat.LAST_ASSISTANT_TURN, args.max_seq_len)
    if args.dry_run:
        dry_run(train_rows, items, rendered, renderer, tokenizer, args)
        return

    key = get_settings().fireworks_api_key
    if not key:
        sys.exit("FIREWORKS_API_KEY is not set (backend/.env or the environment)")
    run_dir = _out() / "serverless"
    run_dir.mkdir(exist_ok=True)
    log_file = (run_dir / "metrics.jsonl").open("a")

    def log(row: dict) -> None:
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        print(json.dumps(row), flush=True)

    service = FiretitanServiceClient(api_key=key, base_url=SERVERLESS_URL)
    enable_serverless_supervised_409_retry(service)

    def score(label: str, path: str) -> None:
        sampler = service.create_sampling_client(model_path=path, tokenizer=tokenizer)
        ask = sampler_ask(sampler, renderer, sdk, get_text_content)
        try:
            results = asyncio.run(
                evaluate(items, None, args.rounds, args.max_tokens, ask, args.concurrency)
            )
        finally:
            getattr(sampler, "close", lambda: None)()
        summary, target = save_card(label, path, results)
        print(json.dumps(summary, indent=1))
        print(f"scorecard: {target}\nsampling at most ${costs(results):.2f}", flush=True)

    try:
        if args.tuned_path:
            score("glimmer-tuned", args.tuned_path)
            return
        client = service.create_lora_training_client(
            base_model=BASE_MODEL, rank=args.rank, alpha=LORA_ALPHA, **LORA_TARGETS
        )
        log({"session": getattr(service, "training_session_id", None)})

        def checkpoint(name: str) -> str:
            path = client.save_weights_for_sampler(name).result().path
            log({"checkpoint": name, "path": path})
            return path

        if not args.no_base:
            score("glimmer-base", checkpoint("wp-base"))  # an adapter of zeros: the base model
        if not args.epochs:
            return
        steps = train(client, datums(rendered, sdk), sdk, args, log, checkpoint)
        tuned = checkpoint("wp-final")
        log({"steps": steps, "tuned": tuned})
        if not args.no_promote:
            promote(service, key, tuned)
        score("glimmer-tuned", tuned)
    finally:
        log_file.close()
        service.close()


def promote(service, key: str, path: str) -> None:
    """A LoRA model from the final adapter, while the session still holds it. Scoring needs no
    model, so a failure here is reported and the run goes on: the checkpoint path stays."""
    from fireworks.training.sdk import FireworksClient

    client = FireworksClient(api_key=key, base_url=GATEWAY_URL)
    try:
        found = [
            r
            for r in client.list_training_session_checkpoints(service.training_session_name)
            if r.get("promotable")
            and "wp-final" in (r.get("checkpointName") or r.get("name") or "")
        ]
        if not found:
            print(f"no promotable checkpoint; the adapter stays at {path}", flush=True)
            return
        target = max(found, key=lambda r: r.get("createTime", ""))
        client.promote_session_checkpoint(
            name=target["name"], output_model_id=OUTPUT_MODEL, base_model=BASE_MODEL
        )
        print(f"promoted {target['name']} to {OUTPUT_MODEL}", flush=True)
    except Exception as e:  # noqa: BLE001 - see the docstring
        print(f"promotion failed ({type(e).__name__}: {str(e)[:200]}); adapter at {path}")
    finally:
        getattr(client, "close", lambda: None)()


if __name__ == "__main__":
    main()
