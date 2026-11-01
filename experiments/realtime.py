"""How fast does one simulated second of the whole brain run, built once and stepped?"""
import sys, time
from pathlib import Path
base = Path(__file__).resolve().parents[1] / 'Drosophila_brain_model'
sys.path.insert(0, str(base))
from brian2 import Network, prefs, ms
from model import create_model, poi, default_params as params
import pandas as pd
neu_sugar = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
             720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
             720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
             720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
             720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
             720575940611875570]
comp = base / '2023_03_23_completeness_630_final.csv'
con = base / '2023_03_23_connectivity_630_final.parquet'
idx = {f: i for i, f in enumerate(pd.read_csv(comp, index_col=0).index)}
t0 = time.time()
neu, syn, mon = create_model(comp, con, params)
pois, neu = poi(neu, [idx[f] for f in neu_sugar], [], params)
net = Network(neu, syn, mon, *pois)
print(f'build: {time.time()-t0:.1f} s')
for chunk in (20, 20, 100, 1000):
    t0 = time.time(); net.run(chunk * ms); dt = time.time() - t0
    print(f'{chunk:5d} ms sim -> {dt:6.2f} s wall  ({dt/(chunk/1000):5.1f}x real time)')
print('spikes:', mon.num_spikes)
