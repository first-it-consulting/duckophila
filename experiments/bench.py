"""Real-time factor of flysim in the closed-loop shape: 20 ms of brain per call, sugar drive on."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from flysim import FlyBrain
from compare_ids import SUGAR

b = FlyBrain(630, seed=0)
b.run(1)  # jit / cache load
b.stimulate({f: 150 for f in SUGAR})
walls = []
for _ in range(500):  # 10 s
    t0 = time.perf_counter(); b.run(20); walls.append(time.perf_counter() - t0)
w = np.array(walls) * 1000
print(f'10 s of brain: {w.sum()/1000:.2f} s wall ({w.sum()/10000:.2f}x real time)')
print(f'per 20 ms step: median {np.median(w):.1f} ms, p99 {np.percentile(w, 99):.1f} ms, max {w.max():.1f} ms; active {b.active}')
