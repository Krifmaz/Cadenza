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
  Gaussian (`gauss`), 1st-order (`ns1`) and 2nd-order (`ef2`) error feedback, Lipshitz,
  Wannamaker 9/3, Gesemann 44.1/48, plus the designed shapers **`ns4`, `ns5`, `ns9`**
  (F-weighted least squares), **`lns15`** (15th order,
  Lipshitz/Wannamaker weighting) and **`ns-adaptive`** (dithered error feedback with an
  adaptive step) – each computed for the actual output rate in both families, with a
  clip-safe limiter so they never run away. **`auto`** picks one per output from the DAC's
  noise floor (see "Auto dither" below).
- **DSD output** (`output_mode = dsd`): 5th- and 7th-order sigma-delta modulators
  **`asdm5`, `asdm7`**, the dithered-quantiser **`dsd5v2`, `dsd7v2`** (lower idle
  tones) and the error-correction loops **`asdm5ec`, `asdm7ec`, `asdm7ec8`** (a look-ahead
  trellis quantiser on the same loops: 4 or 8 bits deep, the cheapest of 4 or 8 candidate
  paths by in-band-weighted error), Schreier-optimised noise-transfer zeros, ≥ 110 dB SNR
  in 20 kHz at DSD64 (measured ≈ 110 / 117 dB plain, 117 / 127 / 128 dB with error
  correction), stable to −0.3 dBFS with an input clipper and a loop guard,
  `dsd_gain_db` headroom (default −3 dB). Rates **DSD64 … DSD1024 and DSD2048** in the
  family of the source (2.8224 … 90.3168 MHz / 3.072 … 98.304 MHz). Packing **DoP**
  (24-bit frames with 0x05/0xFA markers) or **native** (ALSA DSD_U32_BE/LE, DSD_U8; DSD2048
  is native only – DoP cannot carry it). `dsd_oversampling = staged` (long FIR to f_dsd/8
  or /16, short polyphase stage, modulator) or `full` (the long FIR runs to the DSD rate
  itself: a true 1024× PCM stream in 64-bit float that never leaves the pipeline).
- **DSD input**: DSF and DFF (DSDIFF) readers (both bit orders, ID3 / DIIN tags, seeking,
  local files and HTTP), including **DST-compressed DFF** (our own lossless DST decoder,
  bit-exact), **SACD disc images** and **CUE sheets** (see below), feeding either DSD → PCM (CIC decimation + compensating FIR + a
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
  a subformat GUID, via `CADENZA_WASAPI_DSD_GUID` – Windows has no standard one), **ASIO**
  on Windows with native DSD and DoP (feature `asio`; our own loader, no Steinberg SDK, see
  below), **CoreAudio
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
- **CUDA GPU offload (64-bit)** of the long-FIR upsampler and the room convolution
  (feature `cuda`): the upsampler's FFT polyphase core runs on the card – one forward FFT
  per input block, every sub-filter's multiply-accumulate and inverse FFT batched, the
  phases interleaved on the device, only the finished block copied back – and so do the
  time-domain 44.1 ↔ 48 kHz stage and the partitioned room-correction convolver. Our own Stockham FFT kernels compiled at run time
  by the driver's NVRTC, `f64` throughout, filter spectra and frequency-delay lines
  resident on the card. Needs only an NVIDIA driver (≥ 525), no CUDA toolkit – and the
  same binary runs without a GPU (`gpu = auto | on | off | cuda | wgpu`, `gpu_device`; the
  signal path shows `CUDA (RTX 4060) · 64-bit · upsampling + room convolution` or the CPU
  fallback).
- **wgpu GPU offload** (feature `wgpu`) of the same two engines on **AMD, Intel, Apple
  Silicon and NVIDIA** cards through Vulkan, Metal or DX12: the same kernels in WGSL, in
  native `f64` where the card has 64-bit shaders and in double-f32 (two `f32` per value,
  about 48 bits, rounding noise near −270 dB) where it does not (Apple, Intel Arc). Each
  device passes a self-test against the CPU before it is used (`gpu_precision = auto |
  f64 | double-f32`; the signal path shows e.g. `Metal (Apple M2) · double-f32 (≈48-bit)`).
- **`cadenza bench`**: realtime factors (seconds of audio per second) on this machine for
  the chosen filter and rate – the upsampler on the CPU and (row `upsampling (GPU)`) on
  the GPU when present, its FFT engine alone on both, room convolution, and the modulator at every DSD rate (optionally with the long
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
  The same client lives inside the Encore desktop app (`cadenza::embed::EmbeddedPlayer`):
  the engine runs in the app's process with its own `config.toml`, signs in with an app
  token, and Settings → Playback → Cadenza engine in Encore changes every setting live.
- **Signal path** report: every stage with its parameters, as JSON – e.g.
  `FLAC 16/44.1 → 705.6k (sinc-l-1m) → ASDM7 → DSD256 (DoP)`.
- **64-bit maths** everywhere, on the CPU and on the GPU.

## Cadenza vs HQPlayer

An honest comparison with HQPlayer's top configuration. "Matched" means the same spec is
implemented and tested here; the notes say where we differ.

| | HQPlayer | Cadenza | |
|---|---|---|---|
| Upsampling filters | sinc-M / L / Lm, closed-form, poly-sinc-gauss, xtr, ext3, min-phase & apodizing | `sinc`, `sinc-mx`, `sinc-xtr` (8M/16M), `closed-form`, `poly-sinc-gauss` (short/long/xla), `poly-sinc-ext3`; l / m / la / ma variants of each (64 catalogue entries) | matched – ours are our own designs, not copies of HQPlayer's exact responses |
| Filter designer | none (fixed catalogue) | Any window (Kaiser, Dolph–Chebyshev, Gaussian, 3-term cosine), linear or minimum phase, cutoff, transition width and stopband (60–250 dB); length worked out per ratio; response, passband and impulse plots for every filter in the web UI | ahead: HQPlayer has no designer. Tested: every window and phase meets the asked stopband within 1 dB from the asked edge |
| PCM output | up to 1.536 MHz | up to 1.536 MHz (32×), every multiple of both families, rational cross-family | matched |
| Dither / noise shaping | RPDF, TPDF, Gauss1, NS1, NS4, NS5, NS9, LNS15, shaped | rectangular, TPDF, `gauss`, `ns1`, EF2, `ns4`, `ns5`, `ns9`, `lns15`, Lipshitz, Wannamaker, Gesemann, `ns-adaptive`, per output rate; **`auto`** chosen from the DAC's measured noise floor | matched on names (our own designs, not HQPlayer's exact curves) and ahead with `auto`: HQPlayer has no automatic choice. Measured only against a simulated input; not yet with a real DAC and ADC |
| DSD output | DSD64–DSD1024, ASDM5/7 (+ EC, v2/v3 variants) | DSD64–DSD1024 **and DSD2048**, `asdm5`, `asdm7`, `dsd5v2`, `dsd7v2`, error-correction `asdm5ec`, `asdm7ec`, `asdm7ec8` (look-ahead trellis, +6 / +10 dB in-band SNR over the plain loops at DSD64); DoP and native | matched and exceeded on rate; EC matched; no "v3" family yet; EC is CPU-heavy (see `cadenza bench`) |
| 1024× oversampling | yes (GPU recommended) | `dsd_oversampling = full` (long FIR to the DSD rate), CPU or CUDA | matched; very heavy on a CPU – check `cadenza bench --dsd-upsampling --dsd-full` |
| DSD input | DSF/DFF (incl. DST), SACD ISO, DSD→PCM, DSD→DSD via PCM, direct | DSF/DFF including DST-compressed DFF, SACD ISO (stereo and multichannel areas, DST or plain), CUE sheets over PCM or DSD files, gapless DSD → PCM across tracks, re-modulation via PCM, bit-perfect pass-through | matched; tested with synthetic images and files, not yet with a real ripped SACD |
| Convolution | yes | yes, any length, per channel | matched |
| Room measurement | none (REW / Acourate outside) | Built in: sweep from the web UI through any USB microphone, impulse by deconvolution, minimum-phase correction FIR with boost / cut limits, response plots, one click to apply | ahead: HQPlayer has none. Tested against a simulated room (flat within 1.5 dB from 30 to 400 Hz); not yet tried with a real microphone in a real room |
| Headphones | none | Crossfeed (bs2b, Chu Moy, Meier) and KEMAR HRTF at ±30 / 45 / 60° | ahead: HQPlayer has none |
| Parametric EQ | yes (pipeline setup) | 16 biquads global + per matrix output | matched |
| Pipeline matrix / crossovers | yes | N→N up to 32 ch, LR/Butterworth/linear-phase FIR crossovers, delays, polarity, presets | matched |
| Exclusive output | ASIO/WASAPI, CoreAudio hog, ALSA | ASIO (own loader, no SDK), WASAPI exclusive, CoreAudio hog + integer mode, ALSA hw: | matched; ASIO tested against a fake driver only, not yet with a real one |
| Native DSD on Windows | via ASIO | via ASIO (driver switched to DSD, 1-bit bytes in either bit order) and DoP; WASAPI: DoP, opt-in driver GUID | matched in code; not yet tested on a DAC |
| Network endpoint | NAA | Cadenza endpoint (`cadenza endpoint`, mDNS, TCP, ack-paced, PCM + DSD) | matched; our own protocol, not NAA-compatible |
| Multi-room | none | Sync groups of endpoints (`endpoint://a+b+c`): the first is the clock, the others follow its DAC (PCM: adaptive resampling; DSD: word slips) | ahead: HQPlayer has none. In a simulation with DACs ±150 ppm off: within about 1.5 samples at 48 kHz once settled; not yet tried on real DACs over a real LAN |
| GPU offload | CUDA only (64-bit) | CUDA (64-bit, driver only – no toolkit) **and wgpu: Vulkan / Metal / DX12 on AMD, Intel, Apple Silicon and NVIDIA** (64-bit where the card has it, double-f32 ≈48-bit elsewhere); also builds and runs without a GPU; the upsampler's FFT polyphase core, the time-domain 44.1 ↔ 48 kHz stage and the room convolution run on it, verified against the CPU engine (CUDA through a CPU model of the exact kernels; wgpu on real Vulkan in CI with Mesa's software driver: 1.2–2.4e-15 relative in f64, 1–3e-14 in double-f32) and benchmarked (`cadenza bench`, row `upsampling (GPU)`) | exceeded on hardware: HQPlayer has no AMD / Intel / Apple GPU support. Not yet measured on real AMD / Intel / Apple cards, and the CUDA kernels are not yet run on an NVIDIA card since the rational stage was added (none here) |
| 64-bit processing | yes | yes everywhere | matched |
| UPnP / Roon / control | UPnP renderer, Roon via NAA | UPnP/DLNA MediaRenderer (AVTransport, RenderingControl, ConnectionManager, gapless next URI, events) with OpenHome services, HTTP API, web UI, Encore device | matched: UPnP AV and OpenHome (Product, Playlist, Volume, Time, Info) – tested with scripted control points, not yet with BubbleUPnP / Kazoo / mconnect themselves; no Roon |
| AirPlay | no | AirPlay 1 receiver (ALAC, encrypted, volume, title / artist / album, cover art, progress, play / pause / next / previous back to the phone) into the full engine | ahead: HQPlayer has none. Tested with a scripted sender, not yet with an iPhone or Mac; AirPlay 2 is not implemented |
| System-wide input | audio-device input (with a separately installed loopback driver) | Linux: a built-in virtual output device "Cadenza" (PipeWire) and one-app capture before the mixer, following the source rate; Windows: one app or all apps but Cadenza (process loopback); macOS 14.2+: the same through muted process taps, so the sound goes only through Cadenza | ahead on Linux and macOS (nothing extra to install); Windows has capture without a driver but no virtual device. Linux tested against a real headless PipeWire 1.0; the Windows and macOS code is built in CI but not yet run against real audio |
| Loopback verification | none | Bit-perfect check (a random 24-bit pattern found and compared sample for sample in the recording) and THD+N / THD / harmonics / noise of a 997 Hz tone through the chain as configured, from the web UI | ahead: HQPlayer has none. Tested through a software digital loopback (the engine's own output read back); not yet with a real DAC, ADC or S/PDIF loop |
| Licence / price | proprietary | MIT / Apache-2.0, free | – |

### Notes: how Cadenza gets ahead

Where the table says "matched", the plan is to go past HQPlayer, not just tie it. In the order we intend
to do it:

1. **GPU for the upsampler** – done for CUDA and for **wgpu** (AMD, Intel, Apple Silicon, NVIDIA through
   Vulkan / Metal / DX12; HQPlayer is NVIDIA-only): the FFT polyphase stages run on the card (one
   forward FFT per block, all sub-filters in one batched multiply-accumulate and one batched inverse FFT,
   pairs of phases packed into one complex transform), with a seamless switch to the CPU if the device
   fails mid-stream. On wgpu the maths is native `f64` where the card has it and double-f32 (≈48 bits)
   where it does not; every device must pass a self-test against the CPU first. Verified on Mesa's
   software Vulkan driver here and in CI; not yet on a real AMD / Intel / Apple card. The
   time-domain rational stage (44.1 ↔ 48 kHz families, and the first stage of a cascade such as
   44.1 kHz → 1.536 MHz) runs on the GPU too – done on CUDA and wgpu: one workgroup per output
   sample, the history and output clock stay on the host, so a device failure hands the stream to the
   CPU without a gap.
2. **Error-correction modulators** – done on the CPU: `asdm5ec`, `asdm7ec` (4-bit look-ahead, 4 paths)
   and `asdm7ec8` (8 bits, 8 paths, DSD256 and below; above it falls back to `asdm7ec`). A trellis keeps
   the best candidate bit sequences by the low-passed (in-band) quantisation error, which lowers the
   in-band noise by 6 dB (`asdm5ec`) and 10 dB (`asdm7ec`) at DSD64 with the same NTF, guard-free
   at the default gain at every rate; `cadenza bench` lists them per DSD rate. They cost about 2–6× the CPU of the plain loops, `asdm7ec8` 7–20× (on a
   loaded 4-core test VM: `asdm7ec` 3.6× realtime per channel at DSD64, 1.2× at DSD256, 0.5× at DSD512;
   `asdm7ec8` 1.1× at DSD64, 0.3× at DSD256). **A GPU does not help here**, and we measured why:
   every bit depends on the previous one, a stream has only 2 channels × 4–8 trellis paths to share
   out, and cutting the stream into independently modulated time segments (the only way to use a
   GPU's width) wrecks the noise shaping at every seam – SNR at DSD64 falls from 116 dB to 60 dB
   (`asdm7`) and from 127 dB to 56–58 dB (`asdm7ec`), because two runs of the same input never
   converge to the same bits. Three CPU rewrites of the trellis (order-sized state, branch-free
   ranking, paths side by side for vector instructions) were all bit-identical and none was faster
   than the plain code (77 ns/sample for `asdm7ec`, 230 ns for `asdm7ec8` on the test VM; DSD512 in
   real time needs about 37 ns). So the modulator stays one core per channel; the GPU's job is to take
   every filter off the CPU so those cores are free for it. HQPlayer has the same single-core limit.
   DSD2048 is already ours alone.
3. **ASIO on Windows** – done in code: our own runtime loader for the DAC driver's ASIO COM interface (no
   SDK bundled, the FlexASIO approach), PCM in every ASIO sample type, DoP, and native DSD (the driver is
   switched to DSD and fed 1-bit bytes). The host is tested end to end against a fake driver that
   implements the real interface; it has **not yet run against a real ASIO driver or DAC** (no Windows
   machine here). The WASAPI native-DSD GUID stays opt-in (`CADENZA_WASAPI_DSD_GUID`): there is no
   standard one to ship.
4. **UPnP / OpenHome renderer and AirPlay receiver** in the player and the endpoint, so Audirvana,
   mconnect, BubbleUPnP and phones can play into Cadenza. Done: the UPnP/DLNA MediaRenderer, the OpenHome
   services (`[upnp] enabled = true`) and an AirPlay 1 receiver (`[airplay] enabled = true`), see below.
   They run in the player (`cadenza serve`), so everything they receive goes through the DSP; to play
   them on a DAC box, point the player's output at a `cadenza endpoint`. The endpoint itself stays a
   plain output with no receivers of its own. Roon talks to
   HQPlayer only through the closed NAA protocol; Cadenza becomes a Roon zone through Encore's Roon
   extension instead.
5. **DST-compressed DFF and SACD ISO** input, CUE sheets, gapless DSD. DST-compressed DFF is done:
   a DST decoder written from the standard, bit-exact, including the multi-segment frames FFmpeg's
   decoder refuses. It is checked against FFmpeg's decoder (an independent implementation): frames
   from our test encoder decode in FFmpeg exactly like the same DSD stored plain, and in ours bit for
   bit. Not yet tried on DST files from a real SACD authoring tool (none here). Decoding takes
   about 0.1 core for stereo and at most half a core for 5.1 with the longest filters. SACD ISO,
   CUE sheets and gapless DSD are done too (see "SACD images and CUE sheets").
6. **Synchronised multi-room** between Cadenza endpoints – done in code (see "Multi-room" under
   Endpoint). HQPlayer has no multi-room at all. Measured so far only in a simulation on one machine
   (endpoints with simulated DACs 150 ppm fast and 120 ppm slow): the followers stay within about
   1.5 samples (30 µs) of the leader at 48 kHz once settled, 5 samples at most in the first two
   seconds; DSD (DoP) within 6 frames (34 µs). Not yet on real DACs over a real network.
7. **Room correction built in** – done (see "Room correction" below): measure with a USB microphone from
   the web UI (sweep, capture, compute the correction FIR, plots, apply). HQPlayer needs REW or Acourate
   outside the program. Headphone crossfeed and HRTF presets in the engine as well – also done
   (`headphones`). Tested against a simulated room only; not yet with a real microphone.
8. **Filter designer** – done (see "Filter designer" below): pick window, phase, cutoff, transition
   width and stopband depth; Cadenza designs and caches the filter, and plots the response, passband
   ripple and impulse of any filter in the web UI. HQPlayer only offers fixed catalogue entries.
9. **Noise shaping "auto"** – done (see "Auto dither" below): `ns1`, `ns4` and Gaussian dither for
   parity, and `auto`, which picks the shaper per output from the DAC's noise floor, measured from the
   web UI with an input device or taken from the DAC's spec.
10. **System-wide upsampler (standalone, no library, no player)** – Linux done (see "System input"
    below): a PipeWire virtual output device and per-app capture, rate-following, tested against a real
    PipeWire. Windows: per-app and all-apps process loopback; macOS 14.2+: muted process taps, which
    route one app or the whole system through Cadenza like a virtual device, so no HAL plugin is
    needed there. Windows and macOS are built in CI but not yet run against real audio (the CI runners
    have no sound devices). No tray window yet (the web UI is the settings page); no Windows virtual
    device (that needs a signed kernel driver).
    Cadenza stays strictly an upsampler:
    it takes the sound of any app or of the whole system and plays it into the DAC through the engine.
    Two ways in, both with the existing engine settings page in a small tray window and nothing else:
    * *Virtual output device* "Cadenza": pick it as the system or per-app output, Cadenza receives the
      audio and plays it to the real DAC exclusively. macOS: a user-space HAL plugin (the BlackHole
      approach, no kernel extension). Linux: a PipeWire sink. Windows: needs a signed kernel driver, so it
      comes last (attestation signing through the Partner Center).
    * *App capture*: grab one application's audio before the OS mixer touches it. macOS 14.2+ process
      taps (public Core Audio API), Windows 10 2004+ process loopback (`ActivateAudioInterfaceAsync`),
      PipeWire per-stream links on Linux. Available on every OS today without a driver.
    Bit-perfect in: the virtual device follows the source rate (44.1/48 families) and the app's volume at
    100 percent, so the OS has nothing to resample; the signal path shows when that is not the case.
11. **Loopback verification** – done (see "Loopback verification" below): play a test signal, capture it
    from an ADC or a digital loopback, and show measured THD+N and a bit-perfect check. Proves the chain
    instead of trusting it. Tested with a software digital loopback; not yet with real hardware.

Already ahead today: DSD2048, free and open source, built into the Encore desktop app with a full web UI
and automatic device linking, every stage of the chain reported in the signal path.

## Running

```bash
cargo build --release                     # cpal output (default feature)
cargo build --release --features alsa     # + direct ALSA hw: devices, native DSD (Linux)
cargo build --release --features wasapi   # + WASAPI exclusive, DoP (Windows)
cargo build --release --features asio     # + ASIO drivers, native DSD and DoP (Windows)
cargo build --release --features coreaudio # + CoreAudio hog mode, DoP (macOS)
cargo build --release --features cuda     # + CUDA offload (NVIDIA; no toolkit needed to build)
cargo build --release --features wgpu     # + GPU offload on AMD / Intel / Apple / NVIDIA (Vulkan, Metal, DX12)
cargo build --release --features pipewire # + system input: virtual device "Cadenza", app capture (Linux)
cargo build --release --features loopback # + system input: one app, or all apps but Cadenza (Windows 10 2004+)
cargo build --release --features tap      # + system input: one app, or all apps but Cadenza (macOS 14.2+)

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
cadenza render --describe                 # the PCM filter Encore runs on its server (below)
```

The configuration lives in `~/.config/cadenza/config.toml` (platform equivalent via
`dirs`); it is created with defaults on first run. Filter coefficients are cached in
`~/.cache/cadenza` so long designs happen once. Every key below is also a field of
`GET/PUT /api/settings` (the `[endpoint]` table is read by `cadenza endpoint` only).

```toml
[output]
backend = "cpal"        # null | cpal | alsa | wasapi | asio | coreaudio | "endpoint://<name>"
device = ""             # device name or ALSA "hw:1,0"; empty = default
rate = "auto"           # "auto" | "source" | a number such as 352800 (PCM)
bits = 24               # 16 | 24 | 32 (PCM)
max_rate = 1536000      # cap for "auto"

[dsp]
bit_perfect = false
filter = "sinc-l-64k"   # <family>-<phase>-<len>, see `cadenza filters`; or a design, e.g. custom-kaiser-l-c100-t5-s150
attenuation_db = 150.0  # design stopband for the sinc family
dither = "tpdf"         # … ns5 | ns9 | lns15 | ns-adaptive | auto, see `cadenza dithers`
# dac_noise_floor_db = -118.0   # what `auto` chooses from; absent = -120 (measure it in the web UI)
volume_db = 0.0
convolution = ""        # WAV or text impulse response
eq = []                 # [{ kind = "peaking", freq = 100.0, gain_db = -3.0, q = 1.0 }]
eq_enabled = true
# headroom_db = -3.0    # absent = automatic (-3 dB when upsampling, 0 otherwise)
soft_clip = true        # transparent soft clipper as the last guard
headphones = "off"      # off | crossfeed-bs2b | crossfeed-chu-moy | crossfeed-meier | hrtf-30 | hrtf-45 | hrtf-60
output_mode = "pcm"     # pcm | dsd
dsd_rate = "dsd256"     # dsd64 | dsd128 | dsd256 | dsd512 | dsd1024 | dsd2048 (native only)
dsd_modulator = "asdm7" # asdm5 | asdm7 | dsd5v2 | dsd7v2 | asdm5ec | asdm7ec | asdm7ec8
dsd_packing = "dop"     # dop | native
dsd_gain_db = -3.0      # headroom before the modulator
dsd_oversampling = "staged"  # staged | full (long FIR all the way to the DSD rate)
dsd_passthrough = true  # DSF/DFF bits untouched to a DSD output (output_mode = dsd, same rate)
gpu = "auto"            # auto | on | off | cuda | wgpu
gpu_device = 0
gpu_precision = "auto"  # auto | f64 | double-f32 (wgpu only; CUDA is always f64)

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
password = "secret"     # or cookie = "encore_session=…", or token = "…" (an Encore app token, sent as a bearer)
# device_id = ""        # empty = a stable id from hostname and url
# device_name = ""      # empty = "Cadenza on <hostname>"
# embedded = false      # true inside the Encore desktop app (reported to Encore)

[upnp]                  # UPnP / DLNA renderer in `cadenza serve`
enabled = false         # opens a port on the network, so off by default
name = ""               # shown in control apps; empty = "Cadenza (<hostname>)"
port = 49494            # descriptions, control and events (0 = any free port)

[airplay]               # AirPlay 1 receiver in `cadenza serve`
enabled = false         # opens ports on the network, so off by default
name = ""               # shown on iPhones / Macs; empty = "Cadenza (<hostname>)"
port = 5000             # RTSP (0 = any free port); audio uses three more UDP ports
latency_ms = 2000       # buffered before playing (senders schedule 2 s ahead)
volume = true           # follow the sender's volume slider

[endpoint]              # `cadenza endpoint` on the DAC box
name = ""               # advertised name; empty = "Cadenza endpoint on <hostname>"
listen = "0.0.0.0:8778"
backend = ""            # local output backend; empty = platform default
device = ""
buffer_ms = 150         # jitter buffer
mdns = true
```

### `cadenza render` (Encore server-side processing)

Encore's server-side processing (Settings → Playback → Server processing) can run a
downloaded `cadenza` binary instead of the engine compiled into Encore, and updates it from
this repository's releases (Encore → Settings → Download tools). `render` is a pure filter:
interleaved PCM in on stdin, interleaved PCM out on stdout, nothing decoded or encoded.

```bash
ffmpeg -i song.flac -f f64le -acodec pcm_f64le - |
  cadenza render --in-rate 44100 --channels 2 --in-format f64le --rate 192000 \
                 --bits 24 --out-format s32le --set filter=sinc-l-1m --set dither=ns9 \
                 --cache-dir ~/.cache/cadenza |
  ffmpeg -f s32le -ar 192000 -ac 2 -i - -c:a flac -sample_fmt s32 -bits_per_raw_sample 24 out.flac
```

- Input `f32le` or `f64le`; output `s32le` with the 16/24-bit value left-justified (24-bit
  output = value << 8), exactly what Encore's in-process engine produces for the same
  settings, sample for sample. The resampler tail is flushed at EOF.
- `--set key=value` (repeatable) sets any key of the settings schema: `filter`, `dither`,
  `headroom_db` (default 0), `soft_clip`, `headphones`, `dac_noise_floor_db` (for `dither=auto`), `gpu` (auto/on/off, GPU builds), `gpu_device`
  (when a GPU is present) and `gpu_precision` (wgpu builds). Unknown keys are ignored with a warning; a bad value
  exits 2. `--filter`, `--dither`, `--headroom-db` and `--gpu` are aliases.
- `cadenza render --describe` prints one JSON line: name, version, `protocol` (1), input
  and output formats, whether GPU support is compiled in (`gpu`, and `gpu_apis`:
  `["cuda", "wgpu"]`), the filter and dither ids and the
  settings schema (`key`, `label`, `type` select/bool/number, `options`, `default`, `help`,
  `min`/`max`/`step`), so Encore draws the engine's settings from the engine itself.
- stderr is silent except for errors (exit code 2) and, with `--progress`, one
  `{"in_seconds":x}` line per second.

## Releases

Pushing a `v*` tag runs `.github/workflows/release.yml`: it builds the `cadenza` binary for
Linux x86_64 (features `alsa,cuda,wgpu`) and aarch64 (features `alsa,wgpu`) on Ubuntu 22.04 (glibc 2.35,
so it runs on Debian bookworm/trixie and newer), and publishes a GitHub release with
`cadenza-<version>-linux-x86_64.tar.gz`, `cadenza-<version>-linux-aarch64.tar.gz` (each holds
just `cadenza`) and `SHA256SUMS`. Encore downloads the archive for its architecture, checks
it against `SHA256SUMS` and runs `cadenza render --describe` before switching to it.

## UPnP / DLNA renderer

With `[upnp] enabled = true`, `cadenza serve` is a UPnP AV MediaRenderer: control apps
(BubbleUPnP, mconnect, Audirvana, foobar2000, Kazoo in UPnP mode …) find it on the LAN by
SSDP and play into it, and everything goes through the engine – upsampling, DSD, EQ,
matrix, the signal path.

- **AVTransport**: set URI, play, pause, stop, seek (time), next/previous, and
  `SetNextAVTransportURI`, which is queued in the engine so the next track starts
  gaplessly. Title, artist, album, cover and duration come from the DIDL-Lite metadata.
- **RenderingControl**: volume 0–100 (the engine's volume curve) and mute.
- **ConnectionManager**: every format the decoder plays, over HTTP.
- **Events**: subscribers get `LastChange` notifications for transport state, track and
  volume changes.
- Only `http(s)` media URLs are played – never local paths – since anyone on the network
  can send these commands. The renderer has its own port (`[upnp] port`); the control API
  keeps its address.

Tested: SSDP discovery against the running server, and a scripted control point that
drives the whole session end to end – set URI, play, volume, mute, pause, seek, gapless
next URI, events – through a real engine
(`cargo test -p cadenza --test upnp`). Not yet tried with the control apps themselves.

**OpenHome** (Linn's services, used by Kazoo, Lumin, BubbleUPnP's OpenHome mode) runs on
the same device:

- **Playlist**: insert, delete, read, id array with change token, play / pause / stop /
  next / previous, seek by id, index or seconds, repeat and shuffle. The playlist lives
  in Cadenza; the engine is given the current entry and the next, and the one after is
  queued as each track starts – gapless, while the controller edits the list freely.
  Deleting the playing entry plays on with the next one. Up to 1000 entries; `http(s)`
  only.
- **Product**: room name, two sources (Playlist and UPnP AV – whichever a controller uses
  becomes the active one), standby.
- **Volume** (0–100 with mute; no balance or fade), **Time** (track count, duration,
  seconds) and **Info** (track, codec, bit depth, sample rate), all evented.

## Auto dither

Noise shaping trades noise in the band for noise above it. How much is worth it depends on
the output: at 24 bits plain TPDF is already far below any DAC, and at 16 bits the DAC's own
noise decides how hard the shaper needs to work. `dither = "auto"` decides per output:

1. For every candidate – TPDF, `ns1`, `ef2`, `ns4`, `ns5`, `ns9`, `lns15`, gentlest first – it
   works out the audible noise at the output's bits and rate: the quantisation error plus
   dither through the shaper's noise transfer function, weighted by the hearing threshold
   (what the shapers are designed against) and scaled so white noise reads the same as
   A-weighted. Plain A-weighting would rank a psychoacoustic shaper below TPDF, because it
   still counts the noise moved to 15–20 kHz.
2. It takes the first one whose noise is at least 10 dB under the DAC's floor (or the quietest
   when none is). Gentler shapers put less noise above the band, so they win when they are
   already inaudible.

Examples (DAC floor −120 dBFS): 16-bit 44.1 kHz → `lns15`; 16-bit 384 kHz → `ef2`; 24-bit at
any rate → TPDF. The signal path shows the pick, e.g. `Auto (-120 dBFS DAC floor): LNS15 …`.

**The DAC's floor** is `dac_noise_floor_db` (dBFS, A-weighted; −120 when not set). Take it
from the DAC's spec (dynamic range with a minus sign) or **Measure** it in the Noise shaping
card: the engine plays a 997 Hz tone at −20 dBFS and then digital silence (dither off), the
input chosen under Room correction records both, and the A-weighted noise in the silence
relative to the tone gives the floor without knowing the input's gain. Best with a cable from
the DAC's output to an audio interface; a microphone measures the room. The input's own noise
counts too, so the result is an upper bound. API: `POST /api/room/noise {input}`.

Tested: the noise model matches what the quantiser produces (each shaper's weighted noise
relative to TPDF within 1.5 dB of the prediction); the analysis recovers a known floor within
0.5 dB through gain and latency; a measurement through the engine with a simulated input lands
within 1 dB and sets the setting. Not yet tried with a real DAC and ADC.

## Filter designer

The "Filter designer" card in the web UI makes an upsampling filter to measure:

- **Window**: Kaiser, Dolph–Chebyshev (equiripple stopband), Gaussian (no sidelobes, least
  ringing) or Nuttall's 3-term cosine (fixed 71.5 dB).
- **Phase**: linear (symmetric, pre- and post-ringing) or minimum (no pre-ringing).
- **Cutoff**: where the response is −6 dB, 80–100 % of the source Nyquist (below 100 % is
  "apodizing").
- **Transition**: how wide the slope from passband to stopband is, 0.01–50 % of the source
  Nyquist.
- **Stopband**: 60–250 dB (deeper is below what 64-bit coefficients can hold).

Cadenza works out the length at each ratio, so the filter has the same shape in Hz at every
output rate (a 5 % Kaiser at 150 dB is about 7k taps at ×16). The design is a filter id like
the catalogue's – `custom-kaiser-l-c100-t5-s150` – so it is used everywhere a filter id is:
config, `PUT /api/settings`, Encore, and `cadenza render --set filter=…`.

**Plot** shows, for a chosen conversion, the magnitude up to twice the source Nyquist (the
peak level per point, so no sidelobe hides between points), the passband ripple and the
impulse around its peak, with the measured stopband and ripple. **Plot selected filter** does
the same for any catalogue filter (up to 2M taps at that rate). API:
`GET /api/filters/response?id=…&in_rate=…&out_rate=…`.

Tested: every window in both phases meets the asked stopband within 1 dB, measured from the
asked-for stopband edge, with passband ripple under 0.01 dB; doubling the ratio or halving the
transition doubles the length; a design runs through the resampler (including a 44.1 → 192 kHz
cascade) and the whole pipeline. Kaiser's usual length estimate falls short for deep stopbands
(up to 17 dB at 250 dB), so the designer lengthens Kaiser designs by up to 30 %.

Minimum-phase filters up to 256k taps are now converted with 16× padding instead of 4×: at 17k
taps that took the stopband from 3 dB short to on target and the passband ripple from 8e-4 to
2e-5 dB. Minimum-phase filters already in the cache keep their old coefficients until the cache
is cleared.

## Room correction

The "Room correction" card in the web UI measures your speakers and builds the filter:

1. Put a measurement microphone (any USB one the OS sees, e.g. a UMIK-1) at the listening seat,
   pick it, turn the volume down and press **Measure left**, then **Measure right**.
   Each plays a 10 Hz–22 kHz exponential sine sweep from one speaker (6 s, −12 dBFS by
   default) while recording. The engine's EQ, convolution and headphone stages are switched
   off during the sweep and restored after.
2. The recording is divided by the sweep in the frequency domain, which gives the room's
   impulse response. Its response (1/6-octave smoothed) is plotted against the target.
3. The correction is a minimum-phase FIR (16384 taps) that pulls the response to the target
   between 20 Hz and *Correct up to* (500 Hz by default – above that a single microphone
   position says little). It cuts peaks by at most *Max cut* and fills dips by at most *Max
   boost*, never boosts below the speaker's own roll-off, and fades out over half an octave
   at both ends. *Tilt* slopes the target (e.g. −0.5 dB/octave for a warmer house curve).
   The dashed lines show the predicted result.
4. **Apply correction** writes `correction-<time>.wav` into `~/.config/cadenza/room/` and
   loads it as the convolution filter (CPU or GPU, resampled to the stream's rate when it
   differs). **Remove** turns it off. Measurements are kept there as `left.wav` / `right.wav`.

API: `GET /api/room`, `POST /api/room/measure {channel, input, level_db, seconds}`,
`POST /api/room/options {high_hz, max_boost_db, max_cut_db, tilt_db_per_octave, …}`,
`POST /api/room/apply`, `POST /api/room/remove`, `POST /api/room/forget {channel}`.

Tested: in a simulated room with two modes and a dip, the corrected response is flat within
1.5 dB from 30 to 400 Hz, measured through the whole engine (sweep played, captured with
latency and noise, filter applied). Not yet tried with a real microphone in a real room;
there is no microphone calibration file support yet.

## Headphones: crossfeed and HRTF

`headphones` (web UI, config, Encore, and `cadenza render --set headphones=…`) works on
stereo sources, at the source rate before the upsampler:

- **Crossfeed** – Bauer's method as bs2b implements it: each ear also gets the other
  channel low-passed, its own channel slightly high-boosted, and the whole normalised so a
  centred bass note keeps its level. `crossfeed-bs2b` (700 Hz, 4.5 dB), `crossfeed-chu-moy`
  (700 Hz, 6 dB), `crossfeed-meier` (650 Hz, 9.5 dB, the strongest).
- **HRTF** – the two channels played through measured head-related impulse responses of
  loudspeakers at ±30°, ±45° or ±60° (`hrtf-30` …): MIT's KEMAR dummy head (Bill Gardner
  and Keith Martin, MIT Media Lab), the diffuse-field-equalised set, resampled to the
  source rate and scaled so nothing can come out louder than it went in. A dummy head is
  nobody's own head: it moves the sound out of the head for most people, but not
  perfectly in front.

Tested: crossfeed separation at 100 Hz equals the preset's feed within 1 dB, and at
8 kHz it is at least 8 dB more; HRTF puts a left-channel 3 kHz tone at least 5 dB louder
in the left ear at 44.1, 48, 96 and 352.8 kHz; the impulse resampling keeps the response.

## SACD images and CUE sheets

- **SACD disc images** (`.iso` rips): `cadenza play album.iso` (or a path in the API's
  queue) queues every track of the stereo area, titled from the disc text; `album.iso#3`
  is one track, `album.iso#m3` one from the multichannel area. DST and plain discs both
  play, bit-perfect to a DSD output or through the DSD → PCM path. Tracks are cut by the
  frames' time codes, so they join without a gap or an overlap. Images over HTTP work too
  (range requests). Disc text in the Japanese / Chinese / Korean character sets is not
  converted yet.
- **CUE sheets**: `cadenza play album.cue` queues its tracks; `album.cue#3` is one. A
  track runs from its `INDEX 01` to the next one's, cut to the exact sample (FLAC seeks
  land on packet boundaries and are trimmed), so consecutive tracks join sample for
  sample. The audio file can be anything Cadenza plays, DSF/DFF included (cut on whole
  bytes). A sheet that still names the `.wav` it was made for finds the same file name
  with another audio extension. Local files only for now.
- **Gapless DSD**: when the next track is DSD in the same format, the DSD → PCM converter
  runs on into it instead of flushing and starting again, so a live album or a
  continuous work plays without a seam. (Bit-perfect DSD output was already seamless.)

Tested: synthetic SACD images packed the way discs are (DST and plain, several tracks,
seeking anywhere in long tracks), CUE sheets over WAV, FLAC and DSF whose tracks
concatenate to the whole file exactly, and two DSD files that are one recording cut in
two playing to the very same samples as the whole. Not yet tried with real disc rips.

## Desktop app

Cadenza also comes as a desktop app (`crates/cadenza-desktop`, built with Tauri like
Encore's): the engine runs inside it and its control page opens in a native window.

- **Tray icon**: what is playing, Play / Pause, Next, Previous, Show Cadenza, Quit. Closing
  the window keeps the music playing; Quit ends it.
- **Media keys and the OS media controls**: Windows (SMTC), macOS (Now Playing), Linux (MPRIS:
  GNOME, KDE, `playerctl`).
- **One instance**: opening it again brings the window forward. The window's size and
  position are remembered.
- It runs the same services as `cadenza serve` from the same config file (Encore link, UPnP,
  AirPlay, system input, as configured), and serves the control page on `[server] listen`
  when that is free, so a phone can still reach it; otherwise on a free local port.

Installers come with each release: `.deb` and `.AppImage` (Linux), `.exe` and `.msi`
(Windows), `.dmg` (macOS). They are not code-signed, so Windows SmartScreen and macOS
Gatekeeper ask before the first start (on macOS: right-click → Open). To build it yourself:

```sh
cargo install tauri-cli --version 2.12.3 --locked
cd crates/cadenza-desktop
cargo tauri build --features alsa                  # Linux (needs libwebkit2gtk-4.1-dev, libayatana-appindicator3-dev)
cargo tauri build --features wasapi,asio,loopback  # Windows
cargo tauri build --features coreaudio,tap         # macOS
```

Tested on Linux: the window shows the page and plays, MPRIS reports the track and its
Play/Pause reaches the engine, and a second launch hands over to the first without starting
anything. The Windows and macOS builds come from CI; not yet run on those systems.

## Loopback verification

The "Loopback verification" card proves what the chain does instead of trusting the signal
path. Connect an output back to an input (the input chosen under Room correction), set the
volume to 100 % and press **Run**:

- **Bit-perfect check** – the engine is switched to bit-perfect mode and plays 2 s of a
  pseudo-random 24-bit pattern (negated on the right channel). The recording is searched for
  the pattern and compared sample for sample. Only a digital loopback can pass: S/PDIF or AES
  out → in, a DAC's digital monitor, an interface's loopback channel. The verdict says how many
  samples differ and where the first one is, or that the pattern was not found at all (an
  analog path, a level or rate change after Cadenza, the wrong input). The engine's own report
  for that playback is shown next to it.
- **THD+N** – a 997 Hz tone (−3 dBFS by default) through the chain as configured: upsampling,
  dither, room correction and all. Measured with a 7-term Blackman–Harris window (sidelobes
  below −180 dB) over 20 Hz – 20 kHz: THD+N in dB and %, THD (harmonics 2–10), each harmonic,
  noise, the tone's level and frequency. Through an analog loopback that is the DAC and the
  ADC together; through a digital one, Cadenza's own chain.

API: `POST /api/room/verify {input, bit_perfect, thd_n, level_db, seconds}`.

Tested: the analysis on synthetic signals (a pure sine below −150 dB; a known cubic
distortion and noise within 0.2 dB and 1 dB); through the engine with the null output's
recording read back as a perfect digital loopback – bit-perfect confirmed, THD+N below
−125 dB through EQ and 24-bit TPDF, settings restored – and through the same loopback with a
cubic nonlinearity: the pattern is reported not found and the THD matches the distortion within
0.5 dB. Not yet run with a real DAC, ADC or S/PDIF loop.

## System input (Linux, Windows, macOS)

Cadenza as the system's upsampler, with no player of its own: the sound of the desktop or
of one app goes through the engine into the DAC. Build with `--features pipewire` (needs
`libpipewire-0.3-dev` and `clang`); the release binaries don't include it, so they keep
running on machines without PipeWire.

```sh
cadenza serve --system sink           # adds the output device "Cadenza"
cadenza serve --system app:firefox    # takes one app's sound
```

or in config.toml:

```toml
[system]
enabled = true
mode = "sink"        # sink | app
app = ""             # for app: its name, process binary or PipeWire node name
name = "Cadenza"     # the device's name
latency_ms = 150     # buffered before playing
```

- **sink**: a PipeWire node with `media.class = Audio/Sink`. It shows up in the desktop's
  sound settings and every app's output list; pick it for the whole system or one app.
- **app**: a capture stream linked to the app's own playback stream, before the mixer,
  so the app still plays where it did. When the app closes its stream and opens another
  (next video, new tab) the capture follows.
- **Rate**: what arrives starts a live track at its rate, and a new rate starts a new one,
  so the engine upsamples from the source rate. PipeWire runs its graph at one rate; for
  bit-perfect input let it switch to the source's by allowing the rates in
  `~/.config/pipewire/pipewire.conf.d/rates.conf`:
  `context.properties = { default.clock.allowed-rates = [ 44100 48000 88200 96000 176400 192000 ] }`
  and keep the app's volume at 100 %. The signal path shows the rate that arrived
  (`PipeWire 32/44.1 → 24/1411.2`).
- The engine is only taken over when it is idle or already playing the system input; a
  record you started keeps playing. The live stream hands out silence when the system is
  quiet, so the DAC stays open. `GET /api/system` and the web UI show what is arriving.
- Between the capture clock and the DAC clock the buffer absorbs drift by dropping or
  bridging a block when it runs over or dry (as for AirPlay); there is no adaptive
  resampling yet.

Tested against a real headless PipeWire 1.0 with WirePlumber (also in CI): a player sending
to the "Cadenza" device reaches the engine at its level; one app is captured by name and its
second stream followed; with allowed rates set, 44.1 → 48 → 96 kHz sources arrive at their
own rates and the engine follows. Not yet tried on a desktop with a real DAC.

**Windows** (build with `--features loopback`, Windows 10 2004 or later; same flags and
config): process loopback, no driver.

- `app:<name>`: the program's audio (by executable name, `firefox` or `firefox.exe`) and of
  the processes it started, before Windows mixes it with other apps. The program must be
  running when Cadenza starts.
- `sink`: Windows has no virtual device without a signed kernel driver, so this takes *all
  apps except Cadenza*. Make another device the Windows default (or mute it) so the sound is
  not heard twice, and let Cadenza play to the DAC in exclusive mode.
- The audio arrives as 32-bit float at the default device's shared-mode rate, which is what
  Windows mixes the app at anyway: set that device's format to the source rate (Sound →
  device properties → Advanced) for no resampling, and keep the app's volume at 100 %.

Not yet run against real audio: CI builds it and checks the process lookup and formats, but
GitHub's Windows runners have no sound devices.

**macOS** (build with `--features tap`, macOS 14.2 or later; same flags and config): Core
Audio process taps, no driver or plug-in.

- `app:<name>`: the app's processes (by bundle id, its last part, or process name: `Music`,
  `com.apple.Music`, `firefox`). The app must have played sound since it started.
- `sink`: every process except Cadenza.
- The tap is *muted*: what it takes is no longer played to the default output, only through
  Cadenza – so `sink` behaves like a virtual output device, without installing one (which is
  why there is no HAL plug-in).
- It is read through a private aggregate device clocked by the default output device (drift
  compensated) and arrives as 32-bit float at that device's rate: set the rate in Audio MIDI
  Setup to the source's, and keep the app's volume at 100 %. macOS asks once for permission to
  record system audio, for the terminal or app that runs Cadenza.

Not yet run against real audio either: CI builds it on macOS and checks the app matching and
process naming, but the runners have no sound devices.

## AirPlay receiver

With `[airplay] enabled = true`, `cadenza serve` shows up as a speaker on iPhones, iPads,
Macs and in iTunes / Music, and what they play goes through the engine like any other
source: upsampling, DSD, EQ, matrix, the signal path.

- **AirPlay 1** (RAOP): advertised as `_raop._tcp` over mDNS; RTSP on `[airplay] port`;
  Apple Lossless (or 16-bit PCM) over RTP/UDP, 44.1 kHz / 16 bit stereo – that is what
  AirPlay 1 carries.
- **Encryption**: senders only play to receivers that answer their challenge with the
  AirPort Express key, and they wrap each session's AES key with it. That key was pulled
  from the device in 2011 and has been public since; shairport-sync and every other open
  AirPlay 1 receiver embed it, and so does Cadenza. It protects nothing here.
- **Network**: lost packets are asked for again once; one still missing after 250 ms is
  played as silence. The buffer (`latency_ms`) absorbs Wi-Fi jitter. The sender's clock
  and the DAC's drift apart slowly: when the buffer runs a second over, the oldest audio
  is dropped, and an empty buffer is bridged with silence. There is no sample-accurate
  clock sync to the sender yet, so a long session can click once in a while.
- **Control**: the sender's volume slider sets the engine volume (−30…0 dB, and mute;
  `volume = false` leaves it alone); title, artist, album, cover art, duration and
  position show in the web UI and the API (the cover at `/api/live/<id>/cover`). Pause
  and seek on the phone flush the buffer. Play, pause, stop, next and previous in
  Cadenza (web UI, API, Encore) go back to the phone through its remote-control service
  (DACP, found over mDNS), so the phone moves to the next song. A second sender takes
  over from the first, as on an AirPort Express.
- **AirPlay 2** (multi-room from the Home app, AAC, buffered audio) is a different, closed
  protocol and is not implemented. Apple devices have kept playing to AirPlay 1 speakers
  (shairport-sync still lists AirPlay 1 as supported); whether iOS 26 still does is not
  confirmed here.

Tested here: the key (the `Apple-Response` matches OpenSSL byte for byte), RSA-OAEP key
unwrapping, AES-CBC packets, ALAC decoding, the reorder buffer and resend requests, and a
scripted sender that runs a whole encrypted session against a real engine – challenge,
SDP, setup, timing, a lost and resent packet, volume, metadata, cover art (fetched over
the API), Cadenza's transport buttons reaching a stand-in for the phone's remote control,
progress, flush, takeover by a second sender, teardown. **Not yet tried with a real iPhone or Mac** (none here).

## ASIO (Windows)

Build with `--features asio` and set `backend = "asio"`; `device` is the driver's name as
installed (`cadenza devices` lists them), or empty for the first one. No Steinberg SDK is
used: Cadenza reads the installed drivers from `HKLM\SOFTWARE\ASIO`, creates the chosen
one through COM and talks to it through its binary interface (the FlexASIO approach).
Only 64-bit drivers are visible to the 64-bit program.

- **PCM**: the driver's rate is set to the stream's; every ASIO sample type is written
  (16/24/32-bit integers in either byte order, right-aligned 32-bit containers, 32/64-bit
  float – 24-bit audio is exact in all of them).
- **DoP**: on integer channels of 24 bits or more (a float path could alter the markers).
- **Native DSD**: when the driver supports it, it is switched to DSD
  (`kAsioSetIoFormat`), its "sample rate" becomes the DSD bit rate, and every buffer holds
  bytes of 8 one-bit samples in the bit order its channels declare. Silence is the DSD
  idle pattern. `cadenza devices` shows `native dsd_u8` and the top rate the driver
  accepts.
- The driver's reset request (its control panel changed the buffer size, the device was
  replugged) reopens the stream; a driver that stops calling back is reported after 3 s.

Tested here: the whole host – negotiation, buffers, callbacks, sample conversion, native
DSD in both bit orders, pause, reset – against a fake driver implementing the real
interface (`cargo test -p cadenza output::asio`). The Windows loader is compiled in CI.
**Not yet tested with a real ASIO driver or DAC.**

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
  `cadenza bench --dsd-upsampling --dsd-full` first; with a CUDA card the FFT polyphase
  stage of that filter runs on the GPU.
- **DSD sources**: DSF/DFF go to the device bit-perfect when `dsd_passthrough` is on,
  `output_mode = dsd` (or bit-perfect mode) is set, the output rate equals the file's and
  nothing else touches the samples (volume at 0 dB, EQ/convolution/matrix off). With the
  default `output_mode = pcm` – even on a DSD-capable DAC – and in every other case they
  are decimated to 352.8 / 384 kHz PCM and run through the normal chain; DSD → DSD at
  another rate is a re-modulation through PCM, exactly as HQPlayer does it.
- `dsd_gain_db` (default −3 dB) keeps the modulator inside its stable range; the clipper
  and loop guard that protect it count their engagements in the signal path.
- **Error correction** (`asdm5ec`, `asdm7ec`, `asdm7ec8`): the same loops as `asdm5` /
  `asdm7`, but each bit is decided by a look-ahead trellis (an M-algorithm over the next
  4 or 8 bits) that minimises the quantisation error's in-band content instead of its
  instantaneous size. Measured at DSD64, −6 dBFS 1 kHz, 20 kHz band: 110.5 → 116.9 dB
  (`asdm5ec`), 116.8 → 127.3 dB (`asdm7ec`), ≈ 128 dB (`asdm7ec8`); none of the guards
  engages at the default gain at any rate. The bits come out 4 or 8 samples later (a few
  µs). They need several times the CPU of the plain loops – run `cadenza bench` first;
  `asdm7ec8` is meant for DSD256 and below and runs as `asdm7ec` above that.

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

### Multi-room

```bash
cadenza serve --backend "endpoint://Living room+Kitchen+Study"
```

One stream, rendered once, played in step on several endpoints:

- The **first endpoint is the clock**: it plays bit-perfect on its own DAC, as a single
  endpoint does. The sender measures every endpoint's clock offset with pings (the fastest
  recent round trip wins) and fits the leader's DAC timeline from its acknowledgements –
  which sample it presents when.
- Every endpoint **starts on the same instant** (scheduled 0.7 s ahead), and each
  **follower steers onto the leader's timeline**: it measures its own DAC's real rate and
  plays at (leader's rate ÷ its own) through a 48-tap windowed-sinc resampler (better than
  −90 dB; a few hundred ppm at most), with a control loop trimming what is left. Errors over
  10 ms (a seek, a pause, a network stall) are fixed by a jump instead.
- **DSD** cannot be resampled: a DSD follower slips one native word or one DoP frame pair
  (keeping the markers right) when it is more than that off. Each slip is 32 DSD bits;
  it may be audible as a faint tick, a few times a second for a DAC 100 ppm off.
- Followers are therefore **not bit-perfect** (the leader is). Endpoints need this version
  (protocol 2); older ones still work on their own. The DACs' own internal delays are
  not measured: different DAC models may sit a fixed few samples apart.

Tested: three endpoints on one machine with simulated DACs (exact, +150 ppm, −120 ppm),
recording when each sample reached each "DAC" – within about 1.5 samples at 48 kHz once
settled; and a DoP group across a seek – within 6 frames. Clock sync over a real LAN adds
roughly the ping asymmetry (tens of microseconds on wired Ethernet, more on Wi-Fi). Not yet
tried on real DACs.

## GPU offload (CUDA)

Build with `--features cuda`. Nothing links against CUDA: `cudarc` loads `libcuda.so.1` /
`nvcuda.dll` at run time, so the build needs **no CUDA toolkit** and the binary starts on
machines without a GPU. At start-up Cadenza probes the driver and the chosen card and logs
one line: `GPU: CUDA (RTX 4060) · 64-bit available`, or `GPU: CUDA unavailable (…); using
the CPU (64-bit)`. The signal path's `Compute` stage says what the current chain does with
it: `CUDA (RTX 4060) · 64-bit · upsampling + room convolution` (or just one of them) for
the engines that run on the card, `… · available, not used by this chain yet` when nothing
in the chain can use it (no resampling and no room correction). The kernels (`crates/cadenza-dsp/src/gpu/kernels.cu`) are compiled on the user's
machine by the NVRTC that ships with the driver.

Requirements: an NVIDIA card and **driver 525 or newer** (the CUDA 12 driver API; nothing
else to install). Settings in `[dsp]`:

| key          | values                | meaning                                              |
|--------------|-----------------------|------------------------------------------------------|
| `gpu`        | `auto` / `on` / `off` / `cuda` / `wgpu` | `auto` uses a card when present (CUDA first, then wgpu); `on`, `cuda` and `wgpu` warn when they cannot |
| `gpu_device` | `0`, `1`, …           | which card, when there are several                   |
| `gpu_precision` | `auto` / `f64` / `double-f32` | wgpu only, see below                       |

What runs on the GPU:

- **The upsampler's FFT polyphase core** (integer ratios ×2 … ×64 … and small rational
  ones like ×3/2; also the integer stage of a cascade such as 44.1 kHz → 1.536 MHz). The
  `L` sub-filter spectra are uploaded once per stage and shared by the channels; per input
  block the device runs one forward FFT into a frequency-delay line that stays on the
  card, one batched multiply-accumulate over every sub-filter and partition (two real
  sub-filters packed into one complex spectrum, so `L/2` complex products), one batched
  inverse FFT, and interleaves the phases into output order; only that block comes back.
  Delay compensation, decimation and the flush tail are the resampler's, identical for
  both cores, so the stream is the same sample for sample (to f64 rounding: the tests
  compare a CPU model of the exact GPU algorithm with the CPU engine at ~1.5e-15 relative
  for ×2 … ×64, `sinc-l-64k`, `poly-sinc-gauss-l-long` and a 1M-tap filter). Each channel
  has its own CUDA stream. If the card fails mid-stream, the stage moves to the CPU core
  with the last blocks replayed into it – no gap, no click – and the log says so once; a
  failure while building just leaves that stage on the CPU. The device kernels have their
  own tests against that model (`cargo test -p cadenza-dsp --features cuda -- --ignored`),
  which need a card.
- **The time-domain polyphase bank** of the rational conversions (44.1 ↔ 48 kHz families:
  ×160/147, ×147/160 …, and the first stage of a cascade such as 44.1 kHz → 1.536 MHz):
  the `L` sub-filters stay on the card, and per call the new input goes up and one block of
  256 threads computes each output sample (strided partial sums, then a tree reduction).
  The input history and the output clock stay on the host, so a device error just hands
  the same state to the CPU core – no gap, no repeat.
- **Room-correction convolution**: the same partitioned engine (FFT → multiply-accumulate
  over all partitions → inverse FFT).

The sigma-delta modulator stays on the CPU for now (it is sequential). Everything is `double` – no `f32`
anywhere on the CUDA path (the wgpu path is described below). Device memory: an upsampler filter of *T* taps holds about
*16 × T* bytes of shared spectra plus *32 × T / L* bytes of delay line per channel (a
16M-tap filter at ×16: about 256 MiB + 2 × 32 MiB); a room filter of *P* partitions at
block *B* holds *P × B × 32* bytes of spectra plus the same for its delay line.
`cadenza bench` prints the CPU and GPU realtime factors side by side (`upsampling` and
`upsampling (GPU)`).

## GPU offload on AMD, Intel and Apple (wgpu)

Build with `--features wgpu` (needs Rust 1.87). The same polyphase kernels, written in WGSL
(`crates/cadenza-dsp/src/gpu/wgsl.rs`), run through `wgpu` on **Vulkan** (Linux, Windows),
**Metal** (macOS) or **DX12** (Windows). Nothing to install beyond the normal graphics
driver. With `gpu = auto`, CUDA is tried first, then wgpu; `gpu = wgpu` asks for wgpu only.
`gpu_device` counts the adapters wgpu finds (discrete cards first; one entry per card even
when it shows up on both Vulkan and DX12; `cadenza render --describe` lists them).

The arithmetic depends on the card (`gpu_precision`):

- **`f64`** – native 64-bit shaders (`SHADER_F64`: most AMD and NVIDIA cards on Vulkan or
  DX12). The same maths as the CPU; measured 1.7e-15 relative to the CPU engine.
- **`double-f32`** – each value is an unevaluated sum of two `f32` (error-free Knuth /
  Dekker arithmetic), about 48 bits of mantissa: measured 2.5e-14 relative, rounding noise
  near −270 dB, far below any converter. Used where there is no `f64` (Apple Silicon has
  none, Intel Arc neither). It is also the faster choice on most consumer cards, whose
  `f64` units run at 1/16 … 1/64 of their `f32` rate. Not bit-identical to the CPU.
- **`auto`** (default) – `f64` where available, `double-f32` elsewhere.

Every device runs a **self-test** when it is first opened: a small polyphase core on the
device against the CPU model, which must agree to 1e-10. A device that fails (a shader
compiler that "optimised away" the double-f32 error terms, a driver bug) is not used; the
log says why and the CPU runs instead. This is not hypothetical: Mesa's software Vulkan
driver does not fuse `fma`, which is why the exact product uses Dekker's split instead.
Mid-stream device loss moves the stage to the CPU with the history replayed, as on CUDA.

What has been verified, and where: the kernels against the CUDA model in both
arithmetics (`cargo test`), the WGSL compiled to SPIR-V, MSL and HLSL by naga in every
test run, and the whole path – upsampler and room convolution – on real Vulkan with
Mesa's software driver (lavapipe), here and in CI (`CADENZA_WGPU_SOFTWARE=1 cargo test
-p cadenza-dsp --features wgpu -- --ignored wgpu_backend`). The macOS (Metal) and Windows
(DX12) builds compile in CI. **Not yet run on a real AMD, Intel or Apple GPU**, so no speed
figures yet; `cadenza bench` shows them on your machine. Software adapters (lavapipe,
SwiftShader, WARP) are slower than the CPU engine and are skipped unless
`CADENZA_WGPU_SOFTWARE=1` is set.

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
  `config`, `bench`, `endpoint`, `render`), the HTTP API and web UI, and the Encore client.

## Roadmap

- [ ] ASIO output on Windows (native DSD without a driver GUID)
- [x] UPnP/DLNA MediaRenderer and OpenHome services
- [x] AirPlay 1 receiver
- [x] DST-compressed DFF, SACD ISO, CUE sheets, gapless DSD
- [x] Multi-room sync groups of endpoints
- [x] Room measurement and correction from the web UI; headphone crossfeed and HRTF
- [x] Filter designer with response plots
- [x] `ns1`, `ns4`, Gaussian dither and `auto` dither from the DAC's measured noise floor
- [x] System input: PipeWire virtual device and app capture, Windows process loopback, macOS process taps
- [x] Loopback verification: bit-perfect check and THD+N
- [ ] MPRIS, Roon
- [ ] Non-uniformly partitioned convolution for lower latency with long impulses
- [ ] "v3"-class modulators and more poly-sinc variants
- [x] The resampler's FFT polyphase stages on the GPU (CUDA)
- [x] A `wgpu` path for AMD / Intel / Apple GPUs (Vulkan / Metal / DX12)
- [x] The time-domain rational stage on the GPU (CUDA and wgpu)

## Development

```bash
cargo test --release --workspace            # unit + integration tests (null output, no hardware needed)
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p cadenza-dsp --features cuda -- --ignored   # GPU-vs-CPU equality, on a machine with a card
cargo test -p cadenza-dsp --features wgpu -- --ignored wgpu_backend   # the same on any Vulkan/Metal/DX12 GPU
CADENZA_WGPU_SOFTWARE=1 cargo test -p cadenza-dsp --features wgpu -- --ignored wgpu_backend   # …or on lavapipe
```

CI (`.github/workflows/ci.yml`, pull requests and manual dispatch): Ubuntu (fmt, clippy
with and without `cuda,wgpu`, tests with `alsa`, tests with `wgpu` on Mesa's software
Vulkan, build with `cuda`), macOS (build with `coreaudio,wgpu`), Windows (build with
`wasapi,wgpu`). Release builds: `.github/workflows/release.yml` on `v*` tags (see Releases).
