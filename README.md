# Cadenza

**Cadenza** is a free, open-source music playback engine written in Rust. It is a
no-compromise alternative to HQPlayer: bit-perfect output, very long upsampling
filters (up to millions of taps, in real time), dither and noise shaping,
room-correction convolution and a parametric EQ, all computed in 64-bit floating point.

Cadenza is a standalone program (CLI, HTTP control API, built-in web UI) and can also
register itself as a **playback device for the [Encore](https://github.com/Krifmaz/Encore)
music server**: your Encore library shows "Cadenza on <host>" next to your phone and
browser tabs, and you can push a queue to it, control it and see its signal path from
any Encore client.

**Where it runs:** on the machine your DAC is plugged into (a PC, a Mac, a Raspberry Pi
next to the DAC). Cadenza decodes, processes and plays straight into the DAC itself; it
does not hand audio back to anyone. When used with Encore, the Encore server only sends
Cadenza the original file and the commands – all the processing happens on Cadenza's own
hardware, never on the server.

Dual-licensed under MIT or Apache-2.0.

## Features

- **Bit-perfect** path: when nothing needs to touch the samples they go to the device
  untouched, at the source rate and bit depth, and the signal path says so.
- **Upsampling** by 2/4/8/16/32 (44.1 → 352.8 kHz, 48 → 768 kHz…) with windowed-sinc
  FIR filters from 2^8 up to 2^22 taps (configurable transition band and stopband
  attenuation), computed with FFT overlap-save partitioned convolution so even
  multi-million-tap filters run in real time. Linear-phase, **minimum-phase**
  (cepstral method) and **apodizing** variants, like HQPlayer's `-2s` family.
  Rational conversion 44.1 ↔ 48 kHz families (L/M polyphase).
- **Dither & noise shaping** when quantising to 16/24/32-bit: none, rectangular, TPDF,
  2nd-order error feedback, Lipshitz 5-coefficient, Wannamaker 9- and 3-coefficient
  (F-weighted), Gesemann 44.1/48 kHz.
- **Convolution** (room correction) from WAV or text impulse responses, stereo or
  per-channel, any length (uniformly partitioned FFT convolution).
- **Parametric EQ**: up to 16 cascaded biquads (peaking, shelves, pass, notch),
  Audio-EQ-Cookbook formulas.
- **Volume** in dB (64-bit multiply) and mute.
- **Decoding** through Symphonia: FLAC, WAV, AIFF, ALAC, MP3, AAC, Vorbis, in
  WAV/AIFF/FLAC/OGG/MP4/MKV containers, from local files **and HTTP URLs**
  (range requests for seeking, custom headers such as a cookie). Gapless playback.
- **Outputs**: `null` (for tests and headless boxes), `cpal` shared mode on every OS,
  ALSA `hw:` exclusive on Linux (feature `alsa`), WASAPI exclusive on Windows
  (feature `wasapi`). Device listing, chosen device, preferred rate ("auto" picks the
  highest integer multiple of the source family the device supports) and bit depth.
- **Control API** on `127.0.0.1:8777` (JSON + Server-Sent Events) and a single-file
  dark **web UI** showing now playing, transport, volume, the signal path as a chain of
  boxes, and every setting.
- **Encore integration**: logs in, heartbeats as a remote device, receives commands
  instantly over Encore's event stream, streams the original files, and reports plays
  like Encore's own player.
- **Signal path** report: every active stage with its parameters, as JSON.
- **CUDA GPU offload (64-bit)** of the partitioned convolution behind the long-FIR
  upsampler and room correction (feature `cuda`): our own Stockham FFT kernels compiled
  at run time by the driver's NVRTC, `f64` throughout, filter spectra and the
  frequency-delay line resident on the card. Needs only an NVIDIA driver (≥ 525), no CUDA
  toolkit – and the same binary runs without a GPU (`gpu = auto | off | cuda`,
  `gpu_device`; the signal path shows `CUDA (RTX 4060) · 64-bit` or the CPU fallback).

## Running

```bash
cargo build --release                     # cpal output (default feature)
cargo build --release --features alsa     # + direct ALSA hw: devices (Linux)
cargo build --release --features wasapi   # + WASAPI exclusive (Windows)
cargo build --release --features cuda     # + CUDA offload (NVIDIA; no toolkit needed to build)

cadenza devices                           # list output devices and what they support
cadenza filters                           # list upsampling filters
cadenza play song.flac other.flac         # play files (gapless)
cadenza play --bitperfect song.flac       # no DSP at all
cadenza play --filter sinc-l-1m --rate 352800 --dither lipshitz song.flac
cadenza play https://example.org/track.flac
cadenza serve                             # control API + web UI on http://127.0.0.1:8777
cadenza serve --listen 0.0.0.0:8777 --device "hw:1,0" --rate auto
```

The configuration lives in `~/.config/cadenza/config.toml` (platform equivalent via
`dirs`); it is created with defaults on first run. Filter coefficients are cached in
`~/.cache/cadenza` so long designs happen once.

```toml
[output]
backend = "cpal"        # null | cpal | alsa | wasapi
device = ""             # device name or ALSA "hw:1,0"; empty = default
rate = "auto"           # "auto" or a number such as 352800
bits = 24               # 16 | 24 | 32

[dsp]
bit_perfect = false
filter = "sinc-l-64k"   # see `cadenza filters`
dither = "tpdf"         # see `cadenza serve` → /api/dithers
volume_db = 0.0
convolution = ""        # WAV or text impulse response
eq = []                 # [{ kind = "peaking", freq = 100.0, gain_db = -3.0, q = 1.0 }]

[server]
listen = "127.0.0.1:8777"

[encore]
url = "http://192.168.1.10:8080"
username = "me"
password = "secret"     # or cookie = "encore_session=…"
```

## GPU offload (CUDA)

Build with `--features cuda`. Nothing links against CUDA: `cudarc` loads `libcuda.so.1` /
`nvcuda.dll` at run time, so the build needs **no CUDA toolkit** and the binary starts on
machines without a GPU. At start-up Cadenza probes the driver and the chosen card and logs
one line: `GPU: CUDA (RTX 4060) · 64-bit`, or `GPU: CUDA unavailable (…); using the CPU
(64-bit)`. The kernels (`crates/cadenza-dsp/src/gpu/kernels.cu`) are compiled on the user's
machine by the NVRTC that ships with the driver.

Requirements: an NVIDIA card and **driver 525 or newer** (the CUDA 12 driver API; nothing
else to install). Settings in `[dsp]`:

| key          | values                | meaning                                              |
|--------------|-----------------------|------------------------------------------------------|
| `gpu`        | `auto` / `off` / `cuda` | `auto` uses a card when present; `cuda` warns when it cannot |
| `gpu_device` | `0`, `1`, …           | which card, when there are several                   |

What runs on the GPU: the partitioned convolution engine (FFT → multiply-accumulate over
all partitions → inverse FFT, with the frequency-delay line kept on the device) that both
the long-FIR upsampler and room-correction convolution use. Everything is `double` – no
`f32` anywhere on the GPU path. The sigma-delta modulator is sequential and stays on the
CPU. Device memory: a filter of *P* partitions at block *B* holds *P × B × 32* bytes of
spectra plus the same for the delay line (a 16M-tap filter at block 16384: about 1 GiB).

## Architecture

```
                   ┌───────────────────────────── cadenza (binary) ─────────────────────────────┐
                   │                                                                            │
 files / URLs ───▶ │  decode (symphonia, HTTP range reader)  ──▶  Player (queue, gapless,       │
 Encore stream ──▶ │                                              transport, settings)         │
                   │                                                     │                      │
                   │                                                     ▼                      │
                   │        ┌────────────────── cadenza-dsp Pipeline ──────────────────┐        │
                   │        │ volume → EQ → convolution → resampler → dither/quantise │        │
                   │        │   (bit-perfect: every stage bypassed, samples untouched) │        │
                   │        └──────────────────────────────┬───────────────────────────┘        │
                   │                                       ▼                                    │
                   │                   Output trait: null | cpal | alsa hw: | wasapi             │
                   │                                                                            │
                   │  axum control API + SSE ──▶ static/index.html web UI                       │
                   │  encore module: login, heartbeat, command stream, play reporting           │
                   └────────────────────────────────────────────────────────────────────────────┘
```

- `crates/cadenza-dsp` — pure DSP, no I/O: FIR design (Kaiser windowed sinc,
  minimum-phase, apodizing) with a disk cache, polyphase FFT resampler, dither and noise
  shapers, partitioned convolution, biquad EQ, volume, the `Pipeline` and the
  serialisable `SignalPath`.
- `crates/cadenza` — the program: decoding, outputs, the player engine, config and CLI,
  the HTTP API and web UI, and the Encore client.

## Roadmap

- [ ] DSD output (DoP and native) and PCM → DSD modulators
- [ ] Poly-sinc filters with sub-sample optimised passbands, more apodizing variants
- [ ] Non-uniformly partitioned convolution for lower latency with long impulses
- [ ] Multichannel bass management and crossovers
- [ ] CoreAudio hog mode / exclusive on macOS
- [ ] UPnP/OpenHome renderer, MPRIS
- [ ] Network audio endpoint (NAA-style) so the DSP box and the DAC box can differ

## Development

```bash
cargo test                       # unit + integration tests (null output, no hardware needed)
cargo fmt --all -- --check
cargo clippy --all-targets
```
