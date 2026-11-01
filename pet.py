"""Pet the duck in the sim: tell `bridge.py --learn` the duck is being petted, for a few seconds.

    .venv/bin/python -I pet.py            # 3 s of petting
    .venv/bin/python -I pet.py 5

A stand-in for the robot's own petting detector — `pet-detect`, a sound classifier that hears head
scratches on the onboard microphone, inside `robotd`, which today only coos and does not publish the
event. On a real duck this is where that event would come in. The sim has no microphone.
"""
import socket
import sys

PORT = 7897

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
with socket.create_connection(('127.0.0.1', PORT)) as s:
    s.sendall(f'pet {seconds}\n'.encode())
    print(s.makefile().readline().strip())
