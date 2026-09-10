"""Reasoning deltas must be counted as output text, streaming or not.

Backends disagree on the key: sglang streams thoughts as ``reasoning_content``,
vLLM as ``reasoning``. A client that knows only one of them mis-times the other
backend entirely -- the whole thinking phase reads as empty deltas, so TTFT slips
to the first *content* token (or stays 0 for a pure-thinking answer) and every
thinking step's ITL sample is lost.

Upstream owns this behaviour as of v0.5.19 (``_combine_openai_chat_content``);
these tests pin it so a future ``sync_vendored`` bump cannot quietly drop it.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest
from aiohttp import web

from sgl_bench._vendored.sglang import bench_serving

# Each streamed delta is preceded by this much server-side think time, so a
# client that ignores the thinking keys reports a TTFT of ~3x a client that
# doesn't -- far outside timing noise.
STEP = 0.05


def _sse_app(deltas):
    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for delta in deltas:
            await asyncio.sleep(STEP)
            chunk = {"choices": [{"index": 0, "delta": delta}]}
            await resp.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        # Usage-only trailer, as OpenAI-compatible servers emit it.
        await resp.write(
            b"data: "
            + json.dumps({"choices": [], "usage": {"completion_tokens": 7}}).encode()
            + b"\n\n"
        )
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    return app


async def _stream(deltas):
    runner = web.AppRunner(_sse_app(deltas))
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port = runner.addresses[0][:2]
        req = bench_serving.RequestFuncInput(
            prompt="hi",
            api_url=f"http://{host}:{port}/v1/chat/completions",
            prompt_len=1,
            output_len=8,
            model="m",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )
        return await bench_serving.async_request_openai_chat_completions(req)
    finally:
        await runner.cleanup()


async def _json_response(message):
    """Drive the same request func against a non-streaming JSON reply."""

    async def handler(request):
        return web.json_response(
            {
                "choices": [{"index": 0, "message": message}],
                "usage": {"completion_tokens": 7},
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port = runner.addresses[0][:2]
        bench_serving.args.disable_stream = True
        req = bench_serving.RequestFuncInput(
            prompt="hi",
            api_url=f"http://{host}:{port}/v1/chat/completions",
            prompt_len=1,
            output_len=8,
            model="m",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )
        return await bench_serving.async_request_openai_chat_completions(req)
    finally:
        await runner.cleanup()


@pytest.fixture(autouse=True)
def _bench_args(monkeypatch):
    monkeypatch.setattr(
        bench_serving,
        "args",
        SimpleNamespace(
            disable_stream=False,
            disable_ignore_eos=False,
            print_requests=False,
            header=None,
        ),
        raising=False,
    )


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
def test_thinking_phase_is_timed(key):
    """TTFT fires on the first thought, not on the first content token."""
    out = asyncio.run(
        _stream(
            [
                {"role": "assistant"},  # empty opener: not a token
                {key: "think-a"},  # <- first token
                {key: "think-b"},
                {"content": "answer"},
            ]
        )
    )

    assert out.success, out.error
    assert out.generated_text == "think-athink-banswer"
    # The first thought, not the content delta two steps later: ignoring the
    # thinking keys pushes TTFT onto the last delta, i.e. ttft ~= latency.
    assert out.ttft < out.latency - STEP
    # One ITL sample per post-TTFT delta -- thinking steps included.
    assert len(out.itl) == 2
    assert out.output_len == 7  # from usage, unaffected either way


def test_empty_deltas_do_not_start_the_clock():
    """A pure-thinking answer still gets a TTFT (upstream left it at 0.0)."""
    out = asyncio.run(_stream([{"reasoning": "think-only"}]))

    assert out.success, out.error
    assert out.ttft > 0.0
    assert out.itl == []


def test_both_keys_are_not_double_counted():
    """A server echoing the thought under both names must not count it twice."""
    out = asyncio.run(
        _stream([{"reasoning": "think", "reasoning_content": "think"}, {"content": "x"}])
    )

    assert out.success, out.error
    assert out.generated_text == "thinkx"


def test_non_streaming_keeps_the_thoughts():
    """`--disable-stream` must not throw the thinking text away either."""
    out = asyncio.run(_json_response({"reasoning": "think", "content": "answer"}))

    assert out.success, out.error
    assert out.generated_text == "thinkanswer"
    # Non-streaming has no first-token signal: TTFT is defined as the latency.
    assert out.ttft == out.latency
