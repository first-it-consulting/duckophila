"""A steppable re-implementation of the Shiu et al. (2024) whole-brain LIF model.

Brian2 rebuilds and re-launches on every `run()`, which costs ~0.2 s — fine for a 1 s
optogenetics trial, useless for a closed loop that wants 20 ms of brain per sensor frame.
This keeps the same equations, constants, delay and update order, holds the state between
calls, and steps only the neurons that could fire — with numba, on one core.

Per 0.1 ms step, in Brian2's default schedule order:
  1. groups      integrate v, g exactly (linear ODE) for non-refractory neurons
  2. thresholds  v > v_th and not refractory -> spike
  3. synapses    arrivals due now: g += w; new spikes queued D steps ahead;
                 Poisson stimulus: v += w_syn * f_poi
  4. resets      spiking neurons: v = v_rst, g = 0
v and g are `(unless refractory)`, which in Brian2 makes them conditional writes: a refractory
neuron drops synaptic arrivals and Poisson kicks, it does not bank them for later.
**Dormant neurons are not stepped, and that is exact.** Between inputs the dynamics are linear,
so the highest a neuron's v can ever climb with no further input is bounded in closed form:
u + PEAK_G * g above rest (u, g clipped at 0; PEAK_G = max over t of the g -> v kernel, ~0.157).
A non-refractory neuron whose bound stays under threshold cannot spike before its next input, so
it goes dormant: no stepping. Its next synaptic arrival or Poisson kick wakes it, its state is
advanced analytically to that step (the same propagator, raised to the gap), and it carries on.
Under visual drive ~50k neurons sit just off rest and almost none could fire — stepping them all
made the brain 4x slower than real time; this makes cost follow synaptic events instead.

`experiments/compare.py` checks firing rates against the reference model.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

REF = Path(__file__).resolve().parent / 'Drosophila_brain_model'
DATA = {
    630: (REF / '2023_03_23_completeness_630_final.csv', REF / '2023_03_23_connectivity_630_final.parquet'),
    783: (REF / 'Completeness_783.csv', REF / 'Connectivity_783.parquet'),
}

# model.default_params, in ms and mV
DT = 0.1
V_0 = V_RST = -52.0
V_TH = -45.0
T_MBR = 20.0
TAU = 5.0
T_RFC = 2.2
T_DLY = 1.8
W_SYN = 0.275
F_POI = 250.0


# The highest the g -> v kernel climbs: v(t) - V_0 = u0 e^{-t/T} + g0 a (e^{-t/tau} - e^{-t/T}),
# a = tau / (tau - T). For g0 > 0 its second term peaks at PEAK_G * g0. A hair of margin on top.
_tt = np.linspace(0.0, 10 * T_MBR, 200001)
PEAK_G = float(np.max(TAU / (TAU - T_MBR) * (np.exp(-_tt / TAU) - np.exp(-_tt / T_MBR)))) * 1.000001
GAP = V_TH - V_0


@njit(nogil=True, cache=True, inline='always')
def _can_spike(u, g):
    return max(u, 0.0) + PEAK_G * max(g, 0.0) >= GAP


# The propagator raised to every gap up to 10 s, so a wake-up costs two loads, not two exps.
_GAPS = 100_000
E_M = np.exp(-np.arange(_GAPS) * DT / T_MBR)
E_S = np.exp(-np.arange(_GAPS) * DT / TAU)


@njit(nogil=True, cache=True, inline='always')
def _advance(v, g, j, steps):
    """Integrate a dormant neuron `steps` steps in one go — the per-step propagator, raised."""
    if steps <= 0:
        return
    if steps < _GAPS:
        e_m = E_M[steps]
        e_s = E_S[steps]
    else:
        e_m = math.exp(-steps * DT / T_MBR)
        e_s = math.exp(-steps * DT / TAU)
    u = v[j] - V_0
    gj = g[j]
    v[j] = V_0 + u * e_m + gj * (TAU / (TAU - T_MBR)) * (e_s - e_m)
    g[j] = gj * e_s


@njit(nogil=True, cache=True)
def _run(n_steps, t0, v, g, rfc_until, rfc_steps, buf, slot_ids, slot_n, in_slot, t_last,
         act, n_act, is_act, spk, flag, delay, indptr, indices, weights, ev_step, ev_j, poi_kick,
         counts, k_vv, k_vg, k_gg):
    L = buf.shape[0]
    na = n_act[0]
    ptr = 0
    n_ev = ev_step.shape[0]
    for t in range(t0, t0 + n_steps):
        slot = t % L
        row = buf[slot]
        # 1. groups + thresholds + arrivals, for the active set: the threshold reads only v, an
        #    arrival writes only g, so one pass keeps Brian2's order
        na0 = na
        for a in range(na0):
            i = act[a]
            spiked = False
            if t >= rfc_until[i]:
                u = v[i] - V_0
                gi = g[i]
                v[i] = V_0 + u * k_vv + gi * k_vg
                g[i] = gi * k_gg
                spiked = v[i] > V_TH
            flag[a] = spiked
            if row[i] != 0.0:
                if t >= rfc_until[i]:
                    g[i] += row[i]
                row[i] = 0.0
        # 2. arrivals for dormant neurons: catch up to this step, then take the input. Dormant
        #    means non-refractory and unable to have crossed threshold, so nothing is skipped.
        for q in range(slot_n[slot]):
            j = slot_ids[slot, q]
            in_slot[slot, j] = False
            if is_act[j]:
                continue
            _advance(v, g, j, t - t_last[j])
            g[j] += row[j]
            row[j] = 0.0
            t_last[j] = t
            if _can_spike(v[j] - V_0, g[j]):
                is_act[j] = True
                act[na] = j
                na += 1
        slot_n[slot] = 0
        # spikes, in active-set order
        ns = 0
        for a in range(na0):
            if flag[a]:
                spk[ns] = act[a]
                ns += 1
        for b in range(ns):
            s = spk[b]
            rfc_until[s] = t + rfc_steps[s]
            counts[s] += 1
        # 3. synapses: queue this step's spikes D steps ahead; their targets wake when it lands
        os_ = (t + delay) % L
        out = buf[os_]
        for b in range(ns):
            s = spk[b]
            for k in range(indptr[s], indptr[s + 1]):
                j = indices[k]
                out[j] += weights[k]
                if not in_slot[os_, j]:
                    in_slot[os_, j] = True
                    slot_ids[os_, slot_n[os_]] = j
                    slot_n[os_] += 1
        #    the Poisson stimulus: this step's pre-drawn events (`FlyBrain._events`)
        while ptr < n_ev and ev_step[ptr] == t:
            j = ev_j[ptr]
            ptr += 1
            if t >= rfc_until[j]:
                if not is_act[j]:
                    _advance(v, g, j, t - t_last[j])
                    t_last[j] = t
                    is_act[j] = True
                    act[na] = j
                    na += 1
                v[j] += poi_kick
        # 4. resets
        for b in range(ns):
            s = spk[b]
            v[s] = V_RST
            g[s] = 0.0
        # neurons that can no longer fire before their next input go dormant
        keep = 0
        for a in range(na):
            i = act[a]
            if t + 1 >= rfc_until[i] and not _can_spike(v[i] - V_0, g[i]):
                is_act[i] = False
                t_last[i] = t
            else:
                act[keep] = i
                keep += 1
        na = keep
    n_act[0] = na


class FlyBrain:
    """The whole brain, held between calls. Address neurons by FlyWire id."""

    def __init__(self, version: int = 630, seed: int | None = None):
        comp, con = DATA[version]
        ids = pd.read_csv(comp, index_col=0).index.to_numpy()
        self.ids = ids
        self.index = {int(f): i for i, f in enumerate(ids)}
        n = len(ids)

        df = pd.read_parquet(con, columns=['Presynaptic_Index', 'Postsynaptic_Index', 'Excitatory x Connectivity'])
        pre = df['Presynaptic_Index'].to_numpy()
        order = np.argsort(pre, kind='stable')
        self.indices = df['Postsynaptic_Index'].to_numpy()[order].astype(np.int32)
        self.weights = (df['Excitatory x Connectivity'].to_numpy()[order] * W_SYN).astype(np.float64)
        self.indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(pre, minlength=n), out=self.indptr[1:])

        # exact one-step propagator for dv/dt = (v0 - v + g)/T, dg/dt = -g/tau
        self.k_vv = math.exp(-DT / T_MBR)
        self.k_gg = math.exp(-DT / TAU)
        a = TAU / (TAU - T_MBR)
        self.k_vg = a * (self.k_gg - self.k_vv)

        self.delay = round(T_DLY / DT)
        self.v = np.full(n, V_0)
        self.g = np.zeros(n)
        self.rfc_until = np.zeros(n, dtype=np.int64)
        self.rfc_steps = np.full(n, round(T_RFC / DT), dtype=np.int64)
        self.buf = np.zeros((self.delay + 1, n))
        # per ring slot: which neurons have an arrival queued in it, for waking dormant ones
        self.slot_ids = np.zeros((self.delay + 1, n), dtype=np.int64)
        self.slot_n = np.zeros(self.delay + 1, dtype=np.int64)
        self.in_slot = np.zeros((self.delay + 1, n), dtype=np.bool_)
        self.t_last = np.full(n, -1, dtype=np.int64)   # last step a dormant neuron's state reflects
        self.counts = np.zeros(n, dtype=np.int64)
        self.act = np.zeros(n, dtype=np.int64)
        self.n_act = np.zeros(1, dtype=np.int64)
        self.is_act = np.zeros(n, dtype=np.bool_)
        self.spk = np.zeros(n, dtype=np.int64)
        self.flag = np.zeros(n, dtype=np.bool_)
        self.t = 0
        self._poi = {}  # neuron index -> rate Hz
        self._rng = np.random.default_rng(seed)

    @property
    def n(self) -> int:
        return len(self.ids)

    def copy(self) -> 'FlyBrain':
        """An independent brain in the same state — for many trials off one load."""
        b = FlyBrain.__new__(FlyBrain)
        b.__dict__ = {k: (v.copy() if isinstance(v, (np.ndarray, dict)) else v) for k, v in self.__dict__.items()}
        b._rng = np.random.default_rng(self._rng.integers(2 ** 63))
        return b

    @property
    def active(self) -> int:
        """Neurons being stepped — the ones that could fire before their next input."""
        return int(self.n_act[0])

    def stimulate(self, rates: dict[int, float]):
        """Set Poisson drive (Hz) per FlyWire id; 0 removes it. Driven neurons lose their
        refractory period, as in the reference model."""
        for fid, r in rates.items():
            i = self.index[int(fid)]
            if r > 0:
                self._poi[i] = float(r)
                self.rfc_steps[i] = 0
            else:
                self._poi.pop(i, None)
                self.rfc_steps[i] = round(T_RFC / DT)

    def run(self, ms: float) -> np.ndarray:
        """Advance `ms` of brain time. Returns spike counts per neuron for this window."""
        n_steps = round(ms / DT)
        ev_step, ev_j = self._events(n_steps)
        self.counts[:] = 0
        _run(n_steps, self.t, self.v, self.g, self.rfc_until, self.rfc_steps, self.buf,
             self.slot_ids, self.slot_n, self.in_slot, self.t_last,
             self.act, self.n_act, self.is_act, self.spk, self.flag, self.delay, self.indptr, self.indices,
             self.weights, ev_step, ev_j, W_SYN * F_POI, self.counts, self.k_vv, self.k_vg, self.k_gg)
        self.t += n_steps
        return self.counts.copy()

    def _events(self, n_steps: int):
        """This window's Poisson kicks, as (absolute step, neuron) sorted by step.

        The reference draws a Bernoulli(p) per driven neuron per step; that process has geometric
        gaps between events, and it is memoryless, so drawing each window's events afresh is the
        same process — at a cost per event rather than per neuron per step (12k motion neurons
        were 120M draws a second)."""
        if not self._poi:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        idx = np.fromiter(self._poi.keys(), dtype=np.int64, count=len(self._poi))
        p = np.fromiter(self._poi.values(), dtype=np.float64, count=len(self._poi)) * DT / 1000.0
        lam = float(p.max()) * n_steps
        k = int(lam + 6 * math.sqrt(lam) + 4)
        while True:
            steps = np.cumsum(self._rng.geometric(p[:, None], size=(len(p), k)), axis=1) - 1
            if (steps[:, -1] >= n_steps).all():
                break
            k *= 2
        keep = steps < n_steps
        st = steps[keep]
        j = np.broadcast_to(idx[:, None], steps.shape)[keep]
        order = np.argsort(st, kind='stable')
        return (st[order] + self.t).astype(np.int64), j[order].astype(np.int64)

    def rate(self, counts: np.ndarray, fids, ms: float) -> np.ndarray:
        """Firing rate (Hz) of the given FlyWire ids from a `run` result."""
        return np.array([counts[self.index[int(f)]] for f in fids]) * 1000.0 / ms
