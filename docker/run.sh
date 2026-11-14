#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${DEEPFILTERNET_IMAGE:-local/deepfilternet:0.5.6-pytorch2.14.1-cu126}
if [[ ${1:-} == --build ]]; then
  shift
  docker build --file "$repo_dir/docker/Dockerfile" --tag "$image" "$repo_dir/docker"
  if (( $# == 0 )); then exit 0; fi
fi
if (( $# != 2 )); then
  echo "Usage: $0 [--build] INPUT.wav OUTPUT.wav" >&2
  echo 'Optional environment: DEEPFILTERNET_GPU=0, DEEPFILTERNET_CACHE, DEEPFILTERNET_IMAGE' >&2
  exit 2
fi
input=$(realpath -- "$1")
[[ -f $input ]] || { echo 'Input file does not exist' >&2; exit 2; }
mkdir -p -- "$(dirname -- "$2")"
output_dir=$(cd -- "$(dirname -- "$2")" && pwd)
output_name=$(basename -- "$2")
[[ ! -e $output_dir/$output_name ]] || { echo 'Output already exists; choose a new filename' >&2; exit 2; }
cache_dir=${DEEPFILTERNET_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/deepfilternet-docker}
mkdir -p -- "$cache_dir"
cache_dir=$(realpath -- "$cache_dir")
exec docker run --rm --init --gpus "device=${DEEPFILTERNET_GPU:-0}" \
  --user "$(id -u):$(id -g)" --env HOME=/cache \
  --mount "type=bind,src=$repo_dir,dst=/code,readonly" \
  --mount "type=bind,src=$(dirname -- "$input"),dst=/input,readonly" \
  --mount "type=bind,src=$output_dir,dst=/output" \
  --mount "type=bind,src=$cache_dir,dst=/cache" \
  "$image" "/input/$(basename -- "$input")" "/output/$output_name"
