# CUDA Docker runner

This local image extends the official PyTorch CUDA runtime, pinned by digest.
It is not an official upstream DeepFilterNet image. Python dependencies live in
the image; inference imports this checkout through a read-only `/code` mount.
The pip-installed DeepFilterNet package supplies dependencies and is overridden
by `PYTHONPATH=/code/DeepFilterNet` for source imports.

Build:

```bash
./docker/run.sh --build
```

Build and process a recording:

```bash
./docker/run.sh --build /path/to/original.wav /path/to/output/cleaned.wav
```

Reuse the image:

```bash
./docker/run.sh /path/to/original.wav /path/to/output/cleaned.wav
```

Mounts are explicit: checkout → `/code` (read-only), input directory → `/input`
(read-only), output directory → `/output` (writable), model cache → `/cache`
(writable). Output uses the invoking user's UID/GID. Existing output files are
never overwritten. The original recording is not modified.

Set `DEEPFILTERNET_GPU` to the GPU index (default `0`),
`DEEPFILTERNET_CACHE` to the host cache directory, or `DEEPFILTERNET_IMAGE`
to another compatible image tag. CUDA is required; there is no CPU fallback.

The runner loads DeepFilterNet3 and processes 30-second sections with one-second
crossfades. It resamples to the model's 48 kHz rate and back to the original rate,
retains channels, and compensates model delay. It writes PCM16 WAV. A neighboring
`.run.jsonl` records progress, CUDA device, source/script/output hashes, clipping
counts, and final duration/channel validation. An incomplete result has a
`.processing.wav` filename; the final name appears only after checks succeed.
Model weights come from `models/DeepFilterNet3.zip` in this checkout and are
extracted into a cache keyed by the archive SHA-256. Only trusted model checkpoints
should be used: the runtime enables loading their legacy checkpoint format.
