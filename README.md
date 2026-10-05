# Wyoming NeMo Speech

A [Wyoming](https://github.com/OHF-Voice/wyoming) server for Home Assistant that
runs [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) locally. One
process provides speech-to-text with `nemotron-en` and text-to-speech with Magpie
on port 10400. English only.

## Docker

```sh
cp config.example.toml config.toml
docker compose up -d
```

The first start downloads both models into the `nemo-models` volume. The image
is the Vulkan build, which runs on the CPU and on any GPU with a Vulkan driver.
For a GPU, uncomment the matching lines in `compose.yml` and set `device` in
`config.toml`. `docker compose build` builds the image locally.

## Without Docker

Install NeMo-Speech.cpp, then run the server with uv. `--backend` is `cpu`,
`cuda`, or `vulkan`. Models are downloaded on first start.

```sh
curl -fsSL https://github.com/NVIDIA/NeMo-Speech.cpp/raw/v0.2.0/scripts/install.sh |
  sh -s -- --version 0.2.0 --backend cpu
uv sync
uv run wyoming-nemo-speech
```

## Home Assistant

Add the Wyoming Protocol integration with the server's host and port 10400. It
provides a speech-to-text and a text-to-speech engine for an Assist pipeline.
Text-to-speech is streamed, so playback starts on the first sentence of a reply.

To bias recognition toward the names of your areas and exposed entities, create
a long-lived access token in your Home Assistant profile and set
`home_assistant.token` in `config.toml`. Names are fetched at startup and
refreshed once a day.

## Speaker identification

To tell family members apart, record each person and put the files under
`speakers/<Name>/`. Three to five clips of five to ten seconds each are
enough: have the person talk naturally, as if asking the assistant for
something, from where they usually speak to it. Use the same microphone the
assistant hears if you can. Files must be 16 kHz mono 16-bit WAV. Trim leading
and trailing silence. On Linux, this records one eight second clip:

```sh
mkdir -p speakers/Jeremy
arecord -f S16_LE -r 16000 -c 1 -d 8 speakers/Jeremy/1.wav
```

The TitaNet model is downloaded on first start. Then

```sh
uv run wyoming-nemo-speech speakers
```

prints how closely each file matches its speaker and how far the speakers are
from each other. Scores are cosine similarities. Each file should score well
above `speakers.threshold` and each pair of speakers well below it.

When a speaker is recognised, their name is put into the transcript using
`speakers.prefix`, so "turn on the lights" from Jeremy reaches the
conversation agent as "I'm Jeremy. turn on the lights". Describe your family
in the agent's instructions, for example who is a child and what they like,
and the agent can respond accordingly. Prefixed commands are not matched by
Home Assistant's built-in sentences, so an LLM agent should handle them.
Unrecognised voices get the plain transcript. The speaker and score are also
sent in the Wyoming transcript's `context`, which Home Assistant does not read
yet.

## Configuration

The server reads `config.toml` from the working directory. Every key is
optional. See `config.example.toml` for the keys and their defaults.
