"""Reproduce Shiu et al.: sugar GRNs (right) -> MN9 (proboscis motor neuron) spikes."""
import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'Drosophila_brain_model'))
from pathlib import Path
from model import run_exp, default_params as params
import utils as utl

base = Path(__file__).resolve().parents[1] / 'Drosophila_brain_model'
config = dict(path_res=str(base.parent / 'results'),
              path_comp=str(base / '2023_03_23_completeness_630_final.csv'),
              path_con=str(base / '2023_03_23_connectivity_630_final.parquet'),
              n_proc=-1)
neu_sugar = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
             720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
             720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
             720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
             720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
             720575940611875570]
MN9 = 720575940660219265
params['n_run'] = int(sys.argv[1]) if len(sys.argv) > 1 else 6
run_exp(exp_name='sugarR', neu_exc=neu_sugar, params=params, force_overwrite=True, **config)
df = utl.load_exps([f"{config['path_res']}/sugarR.parquet"])
rate, _ = utl.get_rate(df, t_run=params['t_run'], n_run=params['n_run'])
print('active neurons:', len(rate))
print('MN9 rate [Hz]:', rate.loc[MN9, 'sugarR'] if MN9 in rate.index else 0)
