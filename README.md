# Cadenza

**Cadenza** is a free, open-source music playback engine written in Rust. It is a
no-compromise alternative to HQPlayer: bit-perfect output, very long upsampling filters
(up to 16 million taps, in real time), PCM up to 1.536 MHz and DSD up to DSD2048 through
our own sigma-delta modulators, DSD input, dither and high-order noise shaping,
room-correction convolution, parametric EQ, a multichannel pipeline matrix with
crossovers, exclusive output on every OS, a network audio endpoint and optional CUDA
offload – everything computed in 64-bit floating point.

Cadenza is a standalone program (CLI, HTTP control API, built-in web UI) and can also
register itself as a **playback device for the [Encore](https://github.com/Krifmaz/Encore)
music server**: your Encore library shows "Cadenza on <host>" next to your phone and
browser tabs, and you can push a queue to it, control it and see its signal path from
any Encore client.

**Where it runs:** on the machine your DAC is plugged into (a PC, a Mac, a Raspberry Pi
next to the DAC) – or on a strong machine with the DAC on a small **Cadenza endpoint**
box elsewhere on the network. Cadenza decodes, processes and plays straight into the DAC
itself; it does not hand audio back to anyone. When used with Encore, the Encore server
only sends Cadenza the original file and the commands – all the processing happens on
Cadenza's own hardware, never on the server.

Dual-licensed under MIT or Apache-2.0.

## Features

- **Bit-perfect** path: when nothing needs to touch the samples they go to the device
  untouched, at the source rate and bit depth, and the signal path says so. DSF/DFF
  sources pass through bit-perfect to a DoP or native DSD output when `output_mode = dsd`
  (or bit-perfect mode is on) and the rate matches; with the default `output_mode = pcm`
  DSD sources are converted to 352.8 / 384 kHz PCM.
- **Upsampling** to every multiple of the 44.1 and 48 kHz families up to **32× = 1.4112 /
  1.536 MHz PCM** (the hardware limit of today's DACs – 1024× and beyond only exist as
  DSD, see below), including the rational 44.1 ↔ 48 cross-family conversions at every
  ratio (cascaded L/M polyphase). Filter families: windowed `sinc` (2^8 … 2^22 taps,
  Kaiser), `closed-form` (Dolph–Chebyshev), `poly-sinc-gauss` (short / long / xla,
  Gaussian), `sinc-mx` (1M taps at ×16, the transition band in Hz kept constant at every
  ratio, like HQPlayer's sinc-M), `sinc-xtr` (2^23 and 2^24 = 8M / 16M taps, ≥ 200 dB
  stopband, transition ≈ 0.0025 % of the passband) and `poly-sinc-ext3` (short 3-term
  optimised). Each in linear-phase, **minimum-phase** (cepstral method) and **apodizing**
  variants. All run as FFT overlap-save partitioned convolution, channels and polyphase
  phases on all cores. `cadenza filters` / `GET /api/filters` list taps, transition band,
  stopband and latency for any conversion; coefficients are cached on disk in `f64`
  (16M taps = 128 MB).
- **Dither & noise shaping** when quantising to 16/24/32-bit: none, rectangular, TPDF,
  2nd-order error feedback, Lipshitz, Wannamaker 9/3, Gesemann 44.1/48, plus the designed
  shapers **`ns5`, `ns9`** (F-weighted least squares), **`lns15`** (15th order,
  Lipshitz/Wannamaker weighting) and **`ns-adaptive`** (dithered error feedback with an
  adaptive step) – each computed for the actual output rate in both families, with a
  clip-safe limiter so they never run away.
- **DSD output** (`output_mode = dsd`): 5th- and 7th-order sigma-delta modulators
  **`asdm5`, `asdm7`** and the dithered-quantiser **`dsd5v2`, `dsd7v2`** (lower idle
  tones), Schreier-optimised noise-transfer zeros, ≥ 110 dB SNR in 20 kHz at DSD64
  (measured ≈ 112 / 117 dB), stable to −0.3 dBFS with an input clipper and a loop guard,
  `dsd_gain_db` headroom (default −3 dB). Rates **DSD64 … DSD1024 and DSD2048** in the
  family of the source (2.8224 … 90.3168 MHz / 3.072 … 98.304 MHz). Packing **DoP**
  (24-bit frames with 0x05/0xFA markers) or **native** (ALSA DSD_U32_BE/LE, DSD_U8; DSD2048
  is native only – DoP cannot carry it). `dsd_oversampling = staged` (long FIR to f_dsd/8
  or /16, short polyphase stage, modulator) or `full` (the long FIR runs to the DSD rate
  itself: a true 1024× PCM stream in 64-bit float that never leaves the pipeline).
- **DSD input**: DSF and DFF (DSDIFF) readers (both bit orders, ID3 / DIIN tags, seeking,
  local files and HTTP) feeding either DSD → PCM (CIC decimation + compensating FIR + a
  155 dB windowed sinc to 352.8 / 384 kHz; passband ripple < 0.01 dB to 20 kHz), DSD → DSD
  re-modulation at another rate through that PCM path (which is what HQPlayer does too),
  or bit-perfect pass-through.
- **Headroom**: automatic intersample-peak headroom (`headroom_db`, −3 dB when
  upsampling, like HQPlayer), true-peak metering of the processed signal and a transparent
  soft clipper as the last guard – the signal path shows when it ever engaged, with a
  counter.
- **Convolution** (room correction) from WAV or text impulse responses, stereo or
  per-channel, any length (uniformly partitioned FFT convolution, channels in parallel).
- **Parametric EQ**: up to 16 cascaded biquads (peaking, shelves, pass, notch),
  Audio-EQ-Cookbook formulas.
- **Pipeline matrix** (`[matrix]`): N → N routing with a dB gain per cell (up to 32
  channels), per-output chains (crossover, EQ bands, delay in ms and/or samples, polarity,
  gain), crossovers for active speakers (Linkwitz–Riley 12/24/36/48, Butterworth 6…48
  dB/oct, linear-phase FIR with exactly complementary bands), N-way phase alignment so
  3-/4-way sets sum flat, stereo utilities (mono, balance, L/R swap, mid/side) and presets
  `stereo | mono | swap | mid-side | 2.1 | 2-way-active | 3-way-active | custom`. The
  signal path lists the matrix per output channel.
- **Volume** in dB (64-bit multiply) and mute.
- **Decoding** through Symphonia: FLAC, WAV, AIFF, ALAC, MP3, AAC, Vorbis, in
  WAV/AIFF/FLAC/OGG/MP4/MKV containers, plus DSF/DFF, from local files **and HTTP URLs**
  (range requests for seeking, custom headers such as a cookie). Gapless playback.
- **Outputs**: `null` (tests, headless boxes), `cpal` shared mode on every OS, **ALSA
  `hw:`** exclusive on Linux with native DSD and DoP (feature `alsa`), **WASAPI
  exclusive** on Windows with DoP (feature `wasapi`; native DSD only where a driver exposes
  a subformat GUID, via `CADENZA_WASAPI_DSD_GUID` – Windows has no standard one), **CoreAudio
  hog mode** on macOS with device rate/format switching, integer mode where offered and
  DoP (feature `coreaudio`). `cadenza devices` lists rates, bits, channels, DSD support
  (DoP / native layouts / top rate) and exclusive mode. "Auto" rate picks the highest
  multiple of the source family the device supports.
- **Cadenza endpoint** (our NAA): `cadenza endpoint` runs on a small box next to the DAC
  and exposes its output devices; the main instance selects `endpoint://<name>` as its
  backend and streams the final processed PCM or DSD words over TCP (framed protocol:
  hello/caps, open, frames with sequence and presentation time, acks with underrun stats).
  Discovered with mDNS (`_cadenza-endpoint._tcp`), jitter buffer, and resample-free clock
  handling: the sender paces to the DAC's clock from the endpoint's acknowledgements.
- **CUDA GPU offload (64-bit)** of the partitioned convolution engine (feature `cuda`):
  room correction runs on it today (the long-FIR upsampler's own convolution stage is on the
  roadmap, so the signal path marks the GPU active only when the chain really uses it) –
  our own Stockham FFT kernels compiled
  at run time by the driver's NVRTC, `f64` throughout, filter spectra and the
  frequency-delay line resident on the card. Needs only an NVIDIA driver (≥ 525), no CUDA
  toolkit – and the same binary runs without a GPU (`gpu = auto | off | cuda`,
  `gpu_device`; the signal path shows `CUDA (RTX 4060) · 64-bit` or the CPU fallback).
- **`cadenza bench`**: realtime factors (seconds of audio per second) on this machine for
  the chosen filter and rate – upsampler, its FFT engine on the CPU and on the GPU when
  present, room convolution, and the modulator at every DSD rate (optionally with the long
  filter feeding each rate, staged or `--dsd-full`), so you see where your CPU/GPU tops out.
- **Control API** on `127.0.0.1:8777` (JSON + Server-Sent Events) and a single-file
  dark **web UI** showing now playing, transport, volume, the signal path as a chain of
  boxes, and every setting grouped as Output · Upsampling · DSD · Noise shaping · Matrix ·
  GPU · Endpoint · EQ (DSD rates the device cannot take are greyed with "your device stops
  at DSDxxx").
- **Encore integration**: logs in, heartbeats as a remote device (with the full signal
  path, output mode, DSD / headroom / GPU / matrix settings and the device's DSD
  capabilities), receives commands instantly over Encore's event stream, streams the
  original files, and reports plays like Encore's own player.
- **Signal path** report: every stage with its parameters, as JSON – e.g.
  `FLAC 16/44.1 → 705.6k (sinc-l-1m) → ASDM7 → DSD256 (DoP)`.
- **64-bit maths** everywhere, on the CPU and on the GPU.

## Cadenza vs HQPlayer

An honest comparison with HQPlayer's top configuration. "Matched" means the same spec is
implemented and tested here; the notes say where we differ.

| | HQPlayer | Cadenza | |
|---|---|---|---|
| Upsampling filters | sinc-M / L / Lm, closed-form, poly-sinc-gauss, xtr, ext3, min-phase & apodizing | `sinc`, `sinc-mx`, `sinc-xtr` (8M/16M), `closed-form`, `poly-sinc-gauss` (short/long/xla), `poly-sinc-ext3`; l / m / la / ma variants of each (64 catalogue entries) | matched – ours are our own designs, not copies of HQPlayer's exact responses |
| PCM output | up to 1.536 MHz | up to 1.536 MHz (32×), every multiple of both families, rational cross-family | matched |
| Dither / noise shaping | TPDF, NS1–NS9, LNS15 | TPDF, EF2, Lipshitz, Wannamaker, Gesemann, `ns5`, `ns9`, `lns15`, `ns-adaptive`, per output rate | matched (our shaper designs; no NS1–NS4 names) |
| DSD output | DSD64–DSD1024, ASDM5/7 (+ v2/v3 variants) | DSD64–DSD1024 **and DSD2048**, `asdm5`, `asdm7`, `dsd5v2`, `dsd7v2`; DoP and native | matched and exceeded on rate; no "v3" family yet |
| 1024× oversampling | yes (GPU recommended) | `dsd_oversampling = full` (long FIR to the DSD rate), CPU or CUDA | matched; very heavy on a CPU – check `cadenza bench --dsd-upsampling --dsd-full` |
| DSD input | DSF/DFF, DSD→PCM, DSD→DSD via PCM, direct | DSF/DFF, DSD→PCM (CIC + FIR, < 0.01 dB), re-modulation via PCM, bit-perfect pass-through | matched |
| Convolution | yes | yes, any length, per channel | matched |
| Parametric EQ | yes (pipeline setup) | 16 biquads global + per matrix output | matched |
| Pipeline matrix / crossovers | yes | N→N up to 32 ch, LR/Butterworth/linear-phase FIR crossovers, delays, polarity, presets | matched |
| Exclusive output | ASIO/WASAPI, CoreAudio hog, ALSA | WASAPI exclusive, CoreAudio hog + integer mode, ALSA hw: | matched; **no ASIO** |
| Native DSD on Windows | via ASIO | DoP only (no standard WASAPI DSD GUID; opt-in driver GUID) | not matched without ASIO |
| Network endpoint | NAA | Cadenza endpoint (`cadenza endpoint`, mDNS, TCP, ack-paced, PCM + DSD) | matched; our own protocol, not NAA-compatible |
| GPU offload | CUDA (64-bit) | CUDA, 64-bit, driver only – no toolkit; also builds and runs without a GPU; today the room convolution runs on it, verified against the CPU engine and benchmarked (`cadenza bench`) | **partial**: the upsampler's FIR still runs on the CPU |
| 64-bit processing | yes | yes everywhere | matched |
| UPnP / Roon / control | UPnP renderer, Roon via NAA | HTTP API, web UI, Encore device | **not matched**: no UPnP renderer, no Roon |
| Licence / price | proprietary | MIT / Apache-2.0, free | – |

## Running

```bash
cargo build --release                     # cpal output (default feature)
cargo build --release --features alsa     # + direct ALSA hw: devices, native DSD (Linux)
cargo build --release --features wasapi   # + WASAPI exclusive, DoP (Windows)
cargo build --release --features coreaudio # + CoreAudio hog mode, DoP (macOS)
cargo build --release --features cuda     # + CUDA offload (NVIDIA; no toolkit needed to build)

cadenza devices                           # output devices: rates, bits, channels, DSD, exclusive
cadenza filters --in-rate 44100 --out-rate 705600   # taps, transition, stopband, latency
cadenza dithers                           # dither / noise shapers
cadenza play song.flac other.flac         # play files (gapless)
cadenza play --bitperfect song.flac       # no DSP at all
cadenza play --filter sinc-l-1m --rate 352800 --dither lns15 song.flac
cadenza play --output-mode dsd --dsd-rate dsd256 --dsd-modulator asdm7 --dsd-packing dop song.flac
cadenza play --matrix 2.1 --headroom -3 --gpu auto album.dsf
cadenza play https://example.org/track.flac
cadenza bench --filter sinc-l-1m --rate 705600          # realtime factors on this machine
cadenza bench --dsd-upsampling                           # + the long filter feeding each DSD rate
cadenza serve                             # control API + web UI on http://127.0.0.1:8777
cadenza serve --listen 0.0.0.0:8777 --device "hw:1,0" --rate auto
cadenza endpoint --name "Living room"     # on the DAC box; then --backend "endpoint://Living room"
```

The configuration lives in `~/.config/cadenza/config.toml` (platform equivalent via
`dirs`); it is created with defaults on first run. Filter coefficients are cached in
`~/.cache/cadenza` so long designs happen once. Every key below is also a field of
`GET/PUT /api/settings` (the `[endpoint]` table is read by `cadenza endpoint` only).

```toml
[output]
backend = "cpal"        # null | cpal | alsa | wasapi | coreaudio | "endpoint://<name>"
device = ""             # device name or ALSA "hw:1,0"; empty = default
rate = "auto"           # "auto" | "source" | a number such as 352800 (PCM)
bits = 24               # 16 | 24 | 32 (PCM)
max_rate = 1536000      # cap for "auto"

[dsp]
bit_perfect = false
filter = "sinc-l-64k"   # <family>-<phase>-<len>, see `cadenza filters`
attenuation_db = 150.0  # design stopband for the sinc family
dither = "tpdf"         # … ns5 | ns9 | lns15 | ns-adaptive, see `cadenza dithers`
volume_db = 0.0
convolution = ""        # WAV or text impulse response
eq = []                 # [{ kind = "peaking", freq = 100.0, gain_db = -3.0, q = 1.0 }]
eq_enabled = true
# headroom_db = -3.0    # absent = automatic (-3 dB when upsampling, 0 otherwise)
soft_clip = true        # transparent soft clipper as the last guard
output_mode = "pcm"     # pcm | dsd
dsd_rate = "dsd256"     # dsd64 | dsd128 | dsd256 | dsd512 | dsd1024 | dsd2048 (native only)
dsd_modulator = "asdm7" # asdm5 | asdm7 | dsd5v2 | dsd7v2
dsd_packing = "dop"     # dop | native
dsd_gain_db = -3.0      # headroom before the modulator
dsd_oversampling = "staged"  # staged | full (long FIR all the way to the DSD rate)
dsd_passthrough = true  # DSF/DFF bits untouched to a DSD output (output_mode = dsd, same rate)
gpu = "auto"            # auto | off | cuda
gpu_device = 0

[matrix]
enabled = false
preset = "stereo"       # stereo | mono | swap | mid-side | 2.1 | 2-way-active | 3-way-active | custom
[matrix.stereo]
mono = false
balance_db = 0.0        # > 0 attenuates the left channel
swap = false
mid_side = "off"        # off | encode | decode
# [[matrix.outputs]]    # for preset = "custom"
# name = "Sub"
# inputs = [-6.0, -6.0] # dB from each input channel; <= -120 = not connected
# gain_db = 0.0
# delay_ms = 0.0
# delay_samples = 0
# invert = false
# lowpass = { kind = "linkwitz-riley", freq = 80.0, slope_db = 24 }   # butterworth | linear-phase
# eq = [{ kind = "peaking", freq = 40.0, gain_db = 3.0, q = 1.0 }]

[server]
listen = "127.0.0.1:8777"

[encore]
url = "http://192.168.1.10:8080"
username = "me"
password = "secret"     # or cookie = "encore_session=…"

[endpoint]              # `cadenza endpoint` on the DAC box
name = ""               # advertised name; empty = "Cadenza endpoint on <hostname>"
listen = "0.0.0.0:8778"
backend = ""            # local output backend; empty = platform default
device = ""
buffer_ms = 150         # jitter buffer
mdns = true
```

## DSD

Set `output_mode = "dsd"` (or `--output-mode dsd`). PCM is upsampled with the chosen long
filter to f_dsd/8 (DSD64–DSD256) or f_dsd/16 (DSD512 and up), a short 150 dB polyphase
stage takes it to the DSD rate, and the modulator turns it into 1 bit per sample:

```
FLAC 16/44.1 → 705.6k (sinc-l-1m) → ×16 short stage → ASDM7 (-3 dB) → DSD256 (DoP)
```

- **Rates** follow the source family: 44.1 kHz material becomes 2.8224 MHz × n, 48 kHz
  material 3.072 MHz × n. DSD2048 (90.3168 / 98.304 MHz) is offered when the device reports
  it (native only); the UI greys higher rates with "your device stops at DSDxxx" and the
  player steps down to the highest rate the device accepts (native → DoP as a fallback).
- **PCM is capped by hardware at 32× (1.536 MHz)**; 1024× and 2048× only exist as DSD.
- **`dsd_oversampling = full`** runs the long filter all the way to the DSD rate (44.1 kHz
  → 45.1584 MHz for DSD1024) so the modulator sees a true 1024× 64-bit PCM stream. The
  signal path then reports `1024× (full)`. It is very heavy: run
  `cadenza bench --dsd-upsampling --dsd-full` first; the CUDA convolver is the intended home
  for it.
- **DSD sources**: DSF/DFF go to the device bit-perfect when `dsd_passthrough` is on,
  `output_mode = dsd` (or bit-perfect mode) is set, the output rate equals the file's and
  nothing else touches the samples (volume at 0 dB, EQ/convolution/matrix off). With the
  default `output_mode = pcm` – even on a DSD-capable DAC – and in every other case they
  are decimated to 352.8 / 384 kHz PCM and run through the normal chain; DSD → DSD at
  another rate is a re-modulation through PCM, exactly as HQPlayer does it.
- `dsd_gain_db` (default −3 dB) keeps the modulator inside its stable range; the clipper
  and loop guard that protect it count their engagements in the signal path.

## Matrix

Enable `[matrix]` and pick a preset, or describe the outputs yourself. Every output has a
dB gain from each input, an optional crossover (`linkwitz-riley`, `butterworth` or
`linear-phase` FIR), EQ bands, a delay in ms and/or samples, polarity and gain. IIR
crossovers of a multi-way set are phase-aligned against each other so the acoustic sum is
flat; linear-phase FIR bands are exactly complementary and other channels are delayed to
match. The device is opened with the matrix's channel count (`cadenza devices` shows the
maximum each device takes) and the signal path lists one stage per output channel.

## Endpoint

```bash
# on the box next to the DAC
cadenza endpoint --name "Living room" --backend alsa --device hw:1,0
# on the processing machine
cadenza serve --backend "endpoint://Living room"
```

The endpoint is discovered over mDNS and listed by `cadenza devices` / the web UI. The
sender streams the final words (PCM or DSD, DoP or native) and keeps at most half a second
in flight, paced by the endpoint's acknowledgements, which carry its DAC-clock position –
no resampling anywhere, one clock. A second sender is refused while one is playing.

## GPU offload (CUDA)

Build with `--features cuda`. Nothing links against CUDA: `cudarc` loads `libcuda.so.1` /
`nvcuda.dll` at run time, so the build needs **no CUDA toolkit** and the binary starts on
machines without a GPU. At start-up Cadenza probes the driver and the chosen card and logs
one line: `GPU: CUDA (RTX 4060) · 64-bit available`, or `GPU: CUDA unavailable (…); using
the CPU (64-bit)`. The signal path's `Compute` stage says what the current chain does with
it: `CUDA (RTX 4060) · 64-bit · room convolution` when the convolution stage runs on the
card, `… · available, not used by this chain yet` when nothing in the chain can use it yet
(today: a chain without room correction – the upsampler's FIR runs on the CPU). The kernels (`crates/cadenza-dsp/src/gpu/kernels.cu`) are compiled on the user's
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
`cadenza bench` prints the CPU and GPU realtime factors side by side.

## Architecture

```
                   ┌────────────────────────────── cadenza (binary) ──────────────────────────────┐
                   │                                                                              │
 files / URLs ───▶ │  decode (symphonia · DSF/DFF reader · HTTP range)  ──▶  Player (queue,        │
 Encore stream ──▶ │                                                        gapless, settings)    │
                   │                                                             │                │
                   │                                                             ▼                │
                   │  ┌──────────────────────── cadenza-dsp Pipeline ───────────────────────────┐ │
                   │  │ volume → EQ → convolution → matrix → resampler → headroom →             │ │
                   │  │   PCM: dither/quantise   |   DSD: short stage → modulator → DoP/native  │ │
                   │  │   (bit-perfect / DSD pass-through: every stage bypassed)                │ │
                   │  └──────────────────────────────────┬──────────────────────────────────────┘ │
                   │                                     ▼                                        │
                   │   Output: null | cpal | alsa hw: | wasapi | coreaudio | endpoint://name        │
                   │                                                                              │
                   │  axum control API + SSE ──▶ static/index.html web UI                         │
                   │  encore module: login, heartbeat, command stream, play reporting             │
                   │  endpoint module: `cadenza endpoint` server, client output, mDNS            │
                   └──────────────────────────────────────────────────────────────────────────────┘
```

- `crates/cadenza-dsp` — pure DSP, no I/O: FIR design (every filter family, windows,
  minimum-phase, apodizing) with a disk cache, polyphase FFT resampler, dither and noise
  shapers, headroom / true peak / soft clip, partitioned convolution (CPU and CUDA), biquad
  EQ, the matrix and crossovers, the DSD modulators / decimator / packing, volume, the
  `Pipeline`, the serialisable `SignalPath` and `bench`.
- `crates/cadenza` — the program: decoding (Symphonia, DSF/DFF), outputs, the endpoint, the
  player engine, config and CLI (`play`, `serve`, `devices`, `filters`, `dithers`,
  `config`, `bench`, `endpoint`), the HTTP API and web UI, and the Encore client.

## Roadmap

- [ ] ASIO output on Windows (native DSD without a driver GUID)
- [ ] UPnP/OpenHome renderer, MPRIS, Roon
- [ ] Non-uniformly partitioned convolution for lower latency with long impulses
- [ ] "v3"-class modulators and more poly-sinc variants
- [ ] The resampler's own convolution stage on the GPU (today the GPU engine runs the room
      convolution stage, and is reachable through the `BlockConvolver` trait and
      `cadenza bench`)

## Development

```bash
cargo test --release --workspace            # unit + integration tests (null output, no hardware needed)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p cadenza-dsp --features cuda -- --ignored   # GPU-vs-CPU equality, on a machine with a card
```

CI (`.github/workflows/ci.yml`, pull requests and manual dispatch): Ubuntu (fmt, clippy,
tests with `alsa`, build with `cuda`), macOS (build with `coreaudio`), Windows (build with
`wasapi`).
