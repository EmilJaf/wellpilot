"""The serverless training loop, on stand-ins for Fireworks' SDK and the cookbook's renderer:
the real ones live in the cookbook's venv. What the stand-ins cannot show is whether the SDK's
calls still have these shapes; --dry-run and the three-item run in docs/glimmer-sft.md do."""

import asyncio
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from wellpilot import training
from wellpilot.training_serverless import (
    costs,
    datums,
    main,
    parse_args,
    render,
    run,
    sampler_ask,
    train,
)


class Tokens:
    def __init__(self, ints: list[int]) -> None:
        self.ints = ints

    def to_ints(self) -> list[int]:
        return self.ints


class Weights(list):
    def tolist(self) -> list:
        return list(self)


class Renderer:
    def build_supervised_example(self, messages, train_on_what):
        size = len(messages[0]["content"])
        return Tokens(list(range(size + 2))), Weights([0.0] * size + [1.0, 1.0])

    def build_generation_prompt(self, messages):
        self.seen = messages
        return Tokens([1, 2, 3])

    def get_stop_sequences(self):
        return [9]

    def parse_response(self, tokens):
        return {"text": "the reply"}, "stop"


class Future:
    def __init__(self, value) -> None:
        self.value = value

    def result(self):
        return self.value


class Client:
    def __init__(self) -> None:
        self.batches, self.steps = [], 0

    def forward_backward(self, batch, loss):
        assert loss == "cross_entropy"
        self.batches.append(batch)
        return Future(SimpleNamespace(metrics={"loss:sum": 6.0, "response_tokens": 3.0}))

    def optim_step(self, adam, emit_grad_norm_metrics):
        self.steps += 1
        return Future(SimpleNamespace(metrics={"grad_norm": 0.5}))


sdk = SimpleNamespace(
    Datum=lambda model_input, loss_fn_inputs: (model_input, loss_fn_inputs),
    ModelInput=SimpleNamespace(from_ints=list),
    AdamParams=lambda **kw: kw,
    SamplingParams=lambda **kw: kw,
)


def chat(size: int) -> dict:
    return {"messages": [{"role": "user", "content": "x" * size}]}


def test_a_row_over_the_length_stops_the_run_before_anything_is_sent() -> None:
    assert len(render([chat(5)], Renderer(), "last", 7)[0][0]) == 7
    with pytest.raises(SystemExit, match=r"rows \[1\] are over --max-seq-len 6; the longest has 7"):
        render([chat(1), chat(5)], Renderer(), "last", 6)


def test_a_datum_shifts_the_targets_and_keeps_the_answer_weighted() -> None:
    (inputs, target) = datums(render([chat(2)], Renderer(), "last", 9), sdk)[0]
    assert inputs == [0, 1, 2]
    assert target == {"target_tokens": [1, 2, 3], "weights": [0.0, 1.0, 1.0]}


def test_training_covers_every_row_each_epoch_and_saves_each_epoch() -> None:
    client, logged, saved = Client(), [], []
    args = Namespace(epochs=2, batch_size=3, lr=1e-4, seed=0)
    steps = train(client, list(range(7)), sdk, args, logged.append, saved.append)
    assert steps == client.steps == len(client.batches) == 6
    assert sorted(sum(client.batches[:3], [])) == list(range(7))
    assert saved == ["wp-epoch-1", "wp-epoch-2"]
    assert all(len(name) <= 17 for name in saved)
    assert logged[0] == {"step": 1, "epoch": 1, "loss": 2.0, "grad_norm": 0.5}


def test_a_sampler_answers_with_the_tokens_it_read_and_wrote() -> None:
    renderer = Renderer()
    calls = []

    class Sampler:
        def sample(self, prompt, num_samples, sampling_params):
            calls.append(sampling_params)
            return Future(SimpleNamespace(sequences=[SimpleNamespace(tokens=[7, 8])]))

    ask = sampler_ask(Sampler(), renderer, sdk, lambda m: m["text"])
    asked = [{"role": "user", "content": "q"}]
    assert asyncio.run(ask(asked, 50)) == ("the reply", 3, 2)
    assert renderer.seen == asked
    assert calls == [{"max_tokens": 50, "temperature": 0.0, "stop": [9]}]


def item(run: str) -> dict:
    return {
        "run": run,
        "question": "q",
        "as_of": None,
        "evidence": [],
        "gold": None,
        "live_first_accepted": False,
    }


@pytest.fixture
def judged(monkeypatch):
    """The checker sends a first reply back and takes one that follows its reply."""

    def score(entry, answer):
        good = answer == {"fixed": True}
        return {
            "run": entry["run"],
            "valid_json": answer is not None,
            "accepted": good,
            "problems": [] if good else ["undeclared 3"],
        }

    monkeypatch.setattr(training, "score", score)


def test_the_second_round_carries_the_checkers_problems(judged) -> None:
    seen = []

    async def ask(messages, limit):
        seen.append(messages[:])
        return json.dumps({"fixed": len(messages) > 2}), 100, 10

    [result] = asyncio.run(training.evaluate([item("a")], None, 2, 50, ask))
    assert result["accepted_by_round"] == 2 and not result["accepted"]
    assert (result["tokens_in"], result["tokens_out"]) == (200, 20)
    assert json.loads(seen[1][-1]["content"])["problems"] == ["undeclared 3"]
    assert json.loads(result["reply"]) == {"fixed": False}  # the answers stay readable
    assert json.loads(result["last"]["reply"]) == {"fixed": True}
    assert costs([result]) == pytest.approx(200 * 1.96e-6 + 20 * 4.88e-6)


def test_answers_keep_their_order_and_a_failed_one_loses_only_itself(judged) -> None:
    async def ask(messages, limit):
        if len(ask.calls) == 1:
            ask.calls.append(1)
            raise RuntimeError("the pool is busy")
        ask.calls.append(1)
        return '{"fixed": true}', 1, 1

    ask.calls = []
    results = asyncio.run(
        training.evaluate([item("a"), item("b"), item("c")], None, 1, 50, ask, concurrency=2)
    )
    assert [r["run"] for r in results] == ["a", "b", "c"]
    assert [r["accepted"] for r in results] == [True, False, True]


def test_defaults_fit_the_longest_row_and_a_small_batch() -> None:
    args = parse_args([])
    assert args.max_seq_len >= 35222 and args.batch_size == 4 and args.rank == 16


def steps_of(argv: list[str]) -> list[str]:
    done = []

    def checkpoint(name: str) -> str:
        done.append(f"save {name}")
        return name

    run(
        parse_args(argv),
        checkpoint,
        lambda: done.append("train"),
        lambda path: done.append(f"promote {path}"),
        lambda label, path: done.append(f"score {label} {path}"),
        lambda: done.append("release"),
    )
    return done


def test_training_comes_before_any_paid_answer() -> None:
    assert steps_of([]) == [
        "save wp-base",
        "train",
        "save wp-final",
        "promote wp-final",
        "score glimmer-base wp-base",
        "score glimmer-tuned wp-final",
    ]


def test_a_variant_run_trains_scores_only_the_tuned_model_and_keeps_no_model() -> None:
    assert steps_of(["--no-base", "--no-promote", "--tag=-reason"]) == [
        "train",
        "save wp-final",
        "score glimmer-tuned-reason wp-final",
    ]


def test_a_saved_sessions_adapter_is_promoted_under_its_tagged_name(monkeypatch) -> None:
    promoted = {}

    class Client:
        account_id = "acct"

        def __init__(self, api_key, base_url) -> None:
            pass

        def list_training_session_checkpoints(self, name):
            assert name == "accounts/acct/trainingSessions/ts-1"
            return [
                {"name": "other", "checkpointName": "wp-epoch-3", "promotable": True},
                {"name": "final", "checkpointName": "wp-final-abc", "promotable": True},
            ]

        def promote_session_checkpoint(self, name, output_model_id, base_model):
            promoted.update(name=name, model=output_model_id)

    module = SimpleNamespace(FireworksClient=Client)
    monkeypatch.setitem(sys.modules, "fireworks", SimpleNamespace(training=None))
    monkeypatch.setitem(sys.modules, "fireworks.training", SimpleNamespace(sdk=module))
    monkeypatch.setitem(sys.modules, "fireworks.training.sdk", module)
    main(["--promote-session", "ts-1", "--tag=-reason"])
    assert promoted == {"name": "final", "model": "wellpilot-glimmer-writer-reason"}


def test_a_base_only_check_gives_its_slot_back() -> None:
    assert steps_of(["--epochs", "0"]) == [
        "save wp-base",
        "score glimmer-base wp-base",
        "release",
    ]


def test_a_chat_model_is_asked_at_the_agents_reasoning_effort() -> None:
    """Unset, Baseten reasons at xhigh and max_tokens can run out before the answer starts."""
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content="answer")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    llm = SimpleNamespace(client=client, model="m", default_reasoning="medium")
    assert asyncio.run(training._ask(llm, [], 100)) == ("answer", 5, 2)
    assert sent["reasoning_effort"] == "medium"
