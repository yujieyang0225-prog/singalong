#!/usr/bin/env bash
# Run a command in the same environment as the dev container, without VS Code.
#   jan/singalong/run.sh python3 Pulsar/IF_DNR/train.py --epochs 3
#   jan/singalong/run.sh                      # interactive shell
# Builds the image once (your user inside, so files stay yours) and mounts the repo + Innatera's starter repo.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
STARTER="${STARTER:-$HOME/Documents/Hackathon repos/grp-gold-neurotum}"
IMAGE=singalong-dev:4.7.1
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -q -t "$IMAGE" --build-arg REMOTE_USER="$USER" --build-arg REMOTE_UID="$(id -u)" \
    --build-arg REMOTE_GID="$(id -g)" "$HERE/.devcontainer" >/dev/null
fi
TTY=(); [ -t 0 ] && TTY=(-it)
[ $# -eq 0 ] && set -- bash
exec docker run --rm "${TTY[@]}" ${DOCKER_ARGS:-} --group-add innatera \
  -v "$REPO:/workspace" -v "$STARTER:/opt/grp-gold-neurotum:ro" \
  -e PYTHONPATH="/opt/grp-gold-neurotum/inertial-measurement-unit/wristbandgesturerecognition:/workspace/jan/singalong" \
  -e PYTHONWARNINGS=ignore -e MPLCONFIGDIR=/tmp/mpl \
  -w /workspace/jan/singalong "$IMAGE" "$@"
