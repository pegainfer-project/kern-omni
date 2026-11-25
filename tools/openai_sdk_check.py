"""Drives a running pega-omni server with the official `openai` Python SDK.

The SDK is the executable form of OpenAI's API reference: if its stock calls
work unchanged against the server, a client written for OpenAI works too.

    uv run --with openai tools/openai_sdk_check.py --base-url http://127.0.0.1:8000/v1 --model pega-omni-sim

Any engine: voices come from `GET /v1/audio/voices`, `--extra` is passed through.
Pass `--reference-audio` to check a server started with `sim --reference-audio`.
Pass `--x-vector-only` for an engine that accepts only audio-only cloning,
and `--reference-wav PATH` to send an existing PCM16 WAV instead of a tone.
"""

import argparse
import base64
import io
import json
import math
import struct
import wave
from pathlib import Path

import openai


def reference_wav_uri(path: Path | None) -> str:
    if path is not None:
        return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    sample_rate = 16_000
    samples = [round(8_000 * math.sin(math.tau * 440 * i / sample_rate)) for i in range(sample_rate)]
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return "data:audio/wav;base64," + base64.b64encode(data.getvalue()).decode("ascii")


def assert_bad_request(client: openai.OpenAI, model: str, voice: str, param: str, extra_body: dict | None = None) -> None:
    try:
        client.audio.speech.create(model=model, voice=voice, input="x", extra_body=extra_body)
        raise AssertionError(f"invalid {param} accepted")
    except openai.BadRequestError as e:
        assert e.body["param"] == param, e.body


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="pega-omni-sim")
    ap.add_argument("--extra", type=json.loads, default={}, help="the request's `extra` object, as JSON")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--reference-audio", action="store_true", help="exercise both reference modes on sim")
    modes.add_argument("--x-vector-only", action="store_true", help="exercise an audio-only reference engine")
    ap.add_argument("--reference-wav", type=Path, help="reference PCM16 WAV; defaults to a one-second tone")
    args = ap.parse_args()
    reference_audio = args.reference_audio or args.x_vector_only
    if args.reference_wav is not None and not reference_audio:
        ap.error("--reference-wav requires --reference-audio or --x-vector-only")
    client = openai.OpenAI(base_url=args.base_url, api_key="unused")
    voices = client.get("/audio/voices", cast_to=object)["voices"]
    extra = {"extra": args.extra}
    if reference_audio:
        assert voices == [], voices
        voice = "alloy"
        reference = {**extra, "ref_audio": reference_wav_uri(args.reference_wav)}
        icl = {**reference, "ref_text": "A reference transcript."}
        audio_only = {**icl, "x_vector_only_mode": True}
        streamed = audio_only if args.x_vector_only else icl
    else:
        voice = voices[0]
        audio_only = streamed = extra

    wav = client.audio.speech.create(model=args.model, voice=voice, input="Hello from the SDK.", extra_body=audio_only)
    data = wav.read()
    riff, _, wave_tag = struct.unpack("<4sI4s", data[:12])
    assert (riff, wave_tag) == (b"RIFF", b"WAVE") and len(data) > 44, (riff, wave_tag, len(data))

    with client.audio.speech.with_streaming_response.create(
        model=args.model, voice=voice, input="Streaming.", response_format="pcm", extra_body=streamed
    ) as resp:
        pcm = sum(len(c) for c in resp.iter_bytes())
    assert pcm > 0 and pcm % 2 == 0, pcm

    with client.audio.speech.with_streaming_response.create(
        model=args.model, voice=voice, input="Events.", response_format="pcm", stream_format="sse", extra_body=streamed
    ) as resp:
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    audio = sum(len(base64.b64decode(e["audio"])) for e in events if e["type"] == "speech.audio.delta")
    done = events[-1]
    assert audio > 0 and done["type"] == "speech.audio.done" and done["usage"]["output_tokens"] > 0, (audio, done)

    if reference_audio:
        assert_bad_request(client, args.model, voice, "ref_audio", {**extra, "x_vector_only_mode": True})
        assert_bad_request(client, args.model, voice, "ref_audio", {**audio_only, "ref_audio": None})
        malformed = {**audio_only, "ref_audio": "data:audio/wav;base64"}
        assert_bad_request(client, args.model, voice, "ref_audio", malformed)
        if args.x_vector_only:
            for mode in ({}, {"x_vector_only_mode": False}, {"x_vector_only_mode": None}):
                assert_bad_request(client, args.model, voice, "x_vector_only_mode", {**icl, **mode})
        else:
            assert_bad_request(client, args.model, voice, "ref_text", reference)
    else:
        assert_bad_request(client, args.model, "nobody", "voice")

    try:
        client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input="x")
        raise AssertionError("unknown model accepted")
    except openai.NotFoundError as e:
        assert e.body["code"] == "model_not_found", e.body

    assert [m.id for m in client.models.list()] == [args.model]
    reference_result = ", x-vector-only" if args.x_vector_only else ", reference audio" if reference_audio else ""
    print(f"openai {openai.__version__}: wav, pcm stream, sse stream, errors and models{reference_result} all conform")


if __name__ == "__main__":
    main()
