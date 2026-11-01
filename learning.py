"""The mushroom body, made to learn. Stage 4: everything in this file is ours, not the fly's.

**Why a cue gets its Kenyon cells assigned.** In this model the mushroom body has no code of its
own: any input strong enough reaches the same ~3,400 of 5,177 Kenyon cells, whatever it is — four
odours, taste, temperature and humidity overlap 0.98-1.00, and the whole brain runs away to ~450k
spikes/s; a fly uses a sparse, odour-specific 5-10 % (`experiments/mb_check.py`). And the camera
cannot reach it at all. So each cue is given its own random sparse set of Kenyon cells, the way
models of the mushroom body assume random projection-neuron-to-Kenyon-cell wiring (Caron et al.
2013): 10 % of all Kenyon cells at 20 Hz.

**The learning rule** (the connectome has no plasticity; Shiu et al.'s model is static). Dopamine
depresses the Kenyon-cell-to-MBON synapses of the Kenyon cells that were active with it (Hige et
al. 2015; Cohn et al. 2015): after a trial, every KC -> MBON weight is multiplied by
1 - eta * a_kc * d_mbon, where a_kc is whether the Kenyon cell fired and d_mbon how strongly the
dopamine neurons of that MBON's compartment fired, relative to `dan_ref`. *Which* dopamine neurons
belong to an MBON's compartment is the connectome's: the weights of their direct DAN -> MBON synapses.
Weights never change sign and never grow back (no forgetting).

**Valence.** Reward is the PAM cluster, punishment PPL1. An MBON's sign for behaviour is the
opposite of its compartment's dopamine (Aso et al. 2014b; Owald & Waddell 2015): MBONs whose
dopamine input is mostly PAM signal avoidance — reward depresses them, and what is left is approach
— and those under PPL1 signal approach. The compartment is the connectome's (the DAN -> MBON
synapses); the rule is the literature's. (A first version read valence off the MBON's transmitter,
glutamate = avoid: then reward made the paired cue *less* attractive, almost all of it through
MBON09, GABAergic and under PAM. The transmitter rule is too coarse.) The cue's value is the mean
rate of the approach MBONs minus that of the avoidance MBONs while it is shown.

**No nose: the antennal lobe silenced.** The runaway is not the mushroom body's own: just before
the Kenyon cells explode, the antennal lobe's local and projection neurons start firing, and its
recurrent excitation, at one shared synaptic weight, floods the calyx. The duck has no olfactory
input, so for learning the antennal lobe's 1,152 local, projection, input and output neurons are
silenced (`anosmic`, `FlyBrain.silence`): with it, no cue up to 15 % at 30 Hz runs away (0 of 10;
intact, 9 of 10), and MBON responses below the runaway are unchanged. A lesion, and said so.

What remains the connectome's: Kenyon cells -> MBONs -> everything downstream, which MBONs each
dopamine neuron reaches, and the dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import neurons as N


@dataclass
class Params:
    cue_fraction: float = 0.10
    cue_rate: float = 20.0       # Hz on each of a cue's Kenyon cells
    reward_rate: float = 30.0    # Hz on every PAM neuron while petted
    dan_ref: float = 10.0        # Hz of a compartment's dopamine that counts as full
    eta: float = 0.5             # depression per fully paired trial


AL_CLASSES = ('ALLN', 'ALPN', 'ALIN', 'ALON')


def anosmic(brain):
    """The brain with its antennal lobe silenced: what a noseless duck learns with."""
    t = N.table()
    return brain.silence(t.index[t.cell_class.isin(AL_CLASSES)])


class MushroomBody:
    """The Kenyon cells, the MBONs, the dopamine neurons of one `FlyBrain`, and the learning rule on it."""

    def __init__(self, brain, params: Params | None = None, seed: int = 0):
        self.p = params or Params()
        t = N.table().reindex(brain.ids)
        ct = t.cell_type.fillna('')
        self.kc_ids = [int(f) for f in t.index[ct.str.startswith('KC')]]
        self.kc = np.array([brain.index[f] for f in self.kc_ids])
        self.mbon = np.flatnonzero(ct.str.startswith('MBON').to_numpy())
        self.pam = np.flatnonzero(ct.str.startswith('PAM').to_numpy())
        self.ppl1 = np.flatnonzero(ct.str.startswith('PPL1').to_numpy())
        self.rng = np.random.default_rng(seed)
        self.cues: dict[str, np.ndarray] = {}

        pre = np.repeat(np.arange(brain.n), np.diff(brain.indptr))
        is_kc = np.zeros(brain.n, bool); is_kc[self.kc] = True
        mbon_pos = np.full(brain.n, -1); mbon_pos[self.mbon] = np.arange(len(self.mbon))
        self.syn = np.flatnonzero(is_kc[pre] & (mbon_pos[brain.indices] >= 0))   # KC -> MBON synapses
        self.syn_pre = pre[self.syn]
        self.syn_mbon = mbon_pos[brain.indices[self.syn]]
        # each MBON's compartment: its dopamine inputs, weighted by synapse count
        is_dan = np.zeros(brain.n, bool); is_dan[self.pam] = is_dan[self.ppl1] = True
        k = np.flatnonzero(is_dan[pre] & (mbon_pos[brain.indices] >= 0))
        self.dan_w = np.zeros((len(self.mbon), brain.n))
        np.add.at(self.dan_w, (mbon_pos[brain.indices[k]], pre[k]), np.abs(brain.weights[k]))
        self.dan_w /= np.maximum(self.dan_w.sum(1, keepdims=True), 1e-12)
        self.dan_cols = np.flatnonzero(self.dan_w.any(0))
        self.dan_w = self.dan_w[:, self.dan_cols]
        pam_share = self.dan_w[:, np.isin(self.dan_cols, self.pam)].sum(1)
        self.sign = np.where(pam_share > 0.5, -1.0, 1.0)         # under reward: avoid -1; under punishment: approach +1

    def cue(self, name: str) -> dict[int, float]:
        """Rates for a cue's Kenyon cells: a random sparse set, the same set every time it is seen."""
        if name not in self.cues:
            self.cues[name] = self.rng.choice(self.kc_ids, int(self.p.cue_fraction * len(self.kc_ids)), replace=False)
        return {int(f): self.p.cue_rate for f in self.cues[name]}

    def off(self) -> dict[int, float]:
        return {int(f): 0.0 for f in self.kc_ids}

    def reward(self, brain, on: bool = True) -> dict[int, float]:
        return {int(brain.ids[i]): (self.p.reward_rate if on else 0.0) for i in self.pam}

    def value(self, counts: np.ndarray, ms: float) -> float:
        """Approach minus avoidance, mean MBON rate (Hz), over a window."""
        r = counts[self.mbon] * 1000.0 / ms
        app, av = self.sign > 0, self.sign < 0
        return float(r[app].mean() - r[av].mean())

    def learn(self, brain, counts: np.ndarray, ms: float, scale: float = 1.0):
        """The rule, applied after a window: depress KC -> MBON where Kenyon cell and dopamine coincided.
        `scale` is the window's share of a 1 s trial, for a loop that learns every brain step."""
        a = (counts[self.syn_pre] > 0).astype(float)
        dan = counts[self.dan_cols] * 1000.0 / ms
        d = np.clip(self.dan_w @ dan / self.p.dan_ref, 0.0, 1.0)
        brain.weights[self.syn] *= 1.0 - self.p.eta * scale * a * d[self.syn_mbon]


class Learner:
    """The mushroom body in closed loop (`bridge.py --learn`): which cue is in view, whether the duck
    is being petted, the rule applied every brain step, and each cue's value against what it was.

    A cue's value is compared with its own *naive* value — the mean over the first `naive_s` seconds
    it was ever seen — because the two cues start out at different values for no reason but which
    Kenyon cells they drew. The duck approaches a cue whose value has risen by more than `threshold`
    Hz: that gate, and the comparison, are ours. While petting and for `settle_s` after, the value is
    not read: driving every PAM also excites MBONs directly in this model."""

    def __init__(self, brain, mb: MushroomBody, naive_s: float = 8.0, tau_s: float = 1.0,
                 threshold: float = 1.0, settle_s: float = 1.0):
        self.brain, self.mb = brain, mb
        self.naive_s, self.tau_s, self.threshold, self.settle_s = naive_s, tau_s, threshold, settle_s
        self.reward_until = 0.0
        self.seen: dict[str, float] = {}        # seconds in view, valued
        self.naive: dict[str, list] = {}        # [sum, n] over the naive window
        self.value: dict[str, float] = {}       # smoothed value, Hz

    def pet(self, seconds: float, now: float):
        self.reward_until = max(self.reward_until, now + seconds)

    def petting(self, now: float) -> bool:
        return now < self.reward_until

    def rates(self, kind: str | None, now: float) -> dict[int, float]:
        r = self.mb.off()
        if kind:
            r.update(self.mb.cue(kind))
        r.update(self.mb.reward(self.brain, self.petting(now)))
        return r

    def after(self, counts, ms: float, kind: str | None, now: float):
        self.mb.learn(self.brain, counts, ms, scale=ms / 1000.0)
        if kind is None or now < self.reward_until + self.settle_s:
            return
        v = self.mb.value(counts, ms)
        a = 1.0 - np.exp(-ms / 1000.0 / self.tau_s)
        self.value[kind] = v if kind not in self.value else self.value[kind] + a * (v - self.value[kind])
        self.seen[kind] = self.seen.get(kind, 0.0) + ms / 1000.0
        if self.seen[kind] <= self.naive_s:
            acc = self.naive.setdefault(kind, [0.0, 0])
            acc[0] += v; acc[1] += 1

    def learned(self, kind: str) -> float | None:
        """Value now minus naive value, Hz; None until the naive window is complete."""
        if self.seen.get(kind, 0.0) <= self.naive_s:
            return None
        s, n = self.naive[kind]
        return self.value[kind] - s / n

    def approach(self, kind: str | None) -> bool:
        x = self.learned(kind) if kind else None
        return x is not None and x > self.threshold
