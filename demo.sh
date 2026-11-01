#!/bin/sh
# Duckophila, start to finish, in two acts — each recorded to demo/*.rrd (replay: .venv/bin/rerun FILE).
#
#   ./demo.sh            # brings the sim up with the camera if it is not, runs both acts, leaves it up
#   ./demo.sh --no-viz   # without the live viewer (the recordings are still written)
#   ./demo.sh --video    # also record the duck's pose and every brain step, then render demo/duckophila.mp4
#
# Act 1, startle: balls thrown at the duck's face at 1.5 m/s from the centre, the left and the right,
# then a hand at petting speed. The fly's giant fiber should fire for the first three, not the last.
# Act 2, the arena: the duck walks inside a striped drum that turns one way, then the other, every
# 8 s; its head turns with the drum (DNp15), and the body follows the head.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
VIZ=""
VIDEO=""
for a in "$@"; do
    case "$a" in --no-viz) VIZ="--no-viz" ;; --video) VIDEO=1 ;; esac
done
# per act: what the brain did (`--trace`) and where the duck was (`video/record_sim.py`)
trace() { [ -n "$VIDEO" ] && echo "--trace demo/$1_trace.npz" || true; }
record_start() { [ -n "$VIDEO" ] && { "$PY" -I video/record_sim.py "demo/$1_sim.npz" > /dev/null 2>&1 & echo $! > demo/.rec.pid; } || true; }
record_stop() { [ -f demo/.rec.pid ] && { kill "$(cat demo/.rec.pid)" 2>/dev/null; sleep 1; rm -f demo/.rec.pid; } || true; }
mkdir -p "$HERE/demo"
cd "$HERE"

say() { printf '\033[35m~~\033[0m %s  %s\n' "$(date +%H:%M:%S)" "$*"; }
wait_for() { i=0; while [ $i -lt 120 ]; do grep -q "$2" "$1" 2>/dev/null && return 0; sleep 0.5; i=$((i + 1)); done; return 1; }

# Up means answering — a socket file outlives the process that made it. The prop's control port
# is ours (sim/sitecustomize.py), so it also proves the sim was started with the prop in the room.
if ! "$PY" -I -c "import socket,sys; s=socket.socket(); s.settimeout(.5); sys.exit(s.connect_ex(('127.0.0.1', 7899)))"; then
    say "bringing the duck up, with its camera"
    DUCK_SIM_CAMERAS=a sim/duck-sim >/dev/null
fi
"$PY" -I drum.py --off

say "act 1: startle"
# shellcheck disable=SC2046
"$PY" -I bridge.py $VIZ --record demo/startle.rrd $(trace startle) > demo/startle.log 2>&1 &
BRIDGE=$!
wait_for demo/startle.log "wired" || { cat demo/startle.log; exit 1; }
record_start startle
sleep 3
for side in center left right; do
    say "   a ball at 1.5 m/s, $side"
    "$PY" -I throw.py --speed 1.5 --side "$side" >/dev/null
    sleep 3
done
say "   a hand at 0.2 m/s, center — should be ignored"
"$PY" -I throw.py --speed 0.2 >/dev/null
sleep 3
record_stop
kill $BRIDGE 2>/dev/null || true; wait $BRIDGE 2>/dev/null || true
grep STARTLE demo/startle.log | sed 's/^/   /' || true

say "act 2: the arena"
# shellcheck disable=SC2046
"$PY" -I bridge.py --steer body $VIZ --record demo/arena.rrd $(trace arena) > demo/arena.log 2>&1 &
BRIDGE=$!
wait_for demo/arena.log "wired" || { cat demo/arena.log; exit 1; }
record_start arena
"$PY" -I drum.py --speed 30 --period 8 >/dev/null &
DRUM=$!
sleep 33
kill $DRUM 2>/dev/null || true; wait $DRUM 2>/dev/null || true
record_stop
kill $BRIDGE 2>/dev/null || true; wait $BRIDGE 2>/dev/null || true
"$PY" -I drum.py --off
if [ -n "$VIDEO" ]; then
    say "rendering the video"
    "$PY" -I video/make_video.py demo
fi

say "done — recordings: demo/startle.rrd, demo/arena.rrd   (replay: .venv/bin/rerun demo/arena.rrd)"
