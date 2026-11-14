# CUDA Docker runner

This local image extends the official PyTorch CUDA runtime, pinned by digest.
It is not an official upstream DeepFilterNet image. Python dependencies live in
the image; inference imports this checkout through a read-only `/code` mount.
The container's default Python environment is `/opt/global`, exposed through
`VIRTUAL_ENV` and `PATH` and activated by interactive Bash shells. It is analogous
to compai's `~/global`, but uses system site packages to reuse the official image's
PyTorch/CUDA installation. Additional packages install into `/opt/global`; there
is no persistent override of the system Python package protection.

SoundFile/SciPy provide audio I/O when no matching torchaudio wheel is available.
The native `libdf` wheel is compiled from this checkout in a separate build stage.
The native build uses `docker/native/Cargo.toml` and its checked-in lockfile
to compile only `libDF` and `pyDF`, without resolving the GUI demo workspace.
The upstream root lockfile is preserved. Builds enforce `--locked`.

Python inference imports the mounted checkout with `PYTHONPATH=/code/DeepFilterNet`.
Rebuild after changing Rust source or dependency manifests; Python edits are live.

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
to another compatible image tag. Set `DEEPFILTERNET_CONTAINER_NAME` to give
a processing container a stable name for `docker logs -f NAME`. CUDA is required; there is no CPU fallback.

The runner loads DeepFilterNet3 and processes 30-second sections with one-second
crossfades. It resamples to the model's 48 kHz rate and back to the original rate,
retains channels, and compensates model delay. It writes PCM16 WAV. A neighboring
`.run.jsonl` records progress, CUDA device, source/script/output hashes, clipping
counts, and final duration/channel validation. An incomplete result has a
`.processing.wav` filename; the final name appears only after checks succeed.
Model weights come from `models/DeepFilterNet3.zip` in this checkout and are
extracted into a cache keyed by the archive SHA-256. Only trusted model checkpoints
should be used: the runtime enables loading their legacy checkpoint format.
