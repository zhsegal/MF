"""Timing diagnostic for regimprint fits (no model change): ms per gradient and NUTS tree size on the real input.

  python regimprint_timing.py --input inputs.npz [--kind add] [--warmup 150] [--samples 50] [--set k=v ...]
Uses the first tuning fit's data (gene fold 0 held out, locked fold excluded).
"""
import argparse, time
import numpy as np
import jax, jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import initialize_model
from regimprint.config import Config
from regimprint.data import Dataset
from regimprint import prepare as P, fit as F, models

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--kind", default="add")
ap.add_argument("--warmup", type=int, default=150); ap.add_argument("--samples", type=int, default=50)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = Config()
kv = dict(s.split("=", 1) for s in a.set)
cfg = cfg.with_(**{k: type(getattr(cfg, k))(v) for k, v in kv.items()})
print("devices", jax.devices(), "| cfg", {k: getattr(cfg, k) for k in ("k_star", "num_chains", "max_tree_depth", "target_accept")}, flush=True)

t0 = time.time()
ds = Dataset.load(a.input); prep = P.prepare(ds, cfg); gf = prep.gene_fold
tr = np.where((gf != 0) & (gf != cfg.locked_fold))[0]
fs = P.prepare_fold(prep, gene_idx=tr); d = F.to_jax(P.model_data(prep, fs, tr))
print(f"prepare {time.time() - t0:.0f}s; train genes {len(tr)}; B {np.shape(d['B'])}; Mmat {np.shape(d['Mmat'])}", flush=True)

kw = dict(kind=a.kind, p0=cfg.p0, slab=cfg.slab)
info = initialize_model(jax.random.PRNGKey(0), models.m2, model_args=(d,), model_kwargs=kw, init_strategy=init_to_median(num_samples=10))
z = info.param_info.z; n_par = sum(int(np.prod(v.shape)) for v in z.values())
vg = jax.jit(jax.value_and_grad(info.potential_fn))
t0 = time.time(); v, g = vg(z); jax.block_until_ready(g); print(f"params {n_par}; compile+first grad {time.time() - t0:.1f}s; U={float(v):.1f}", flush=True)
t0 = time.time()
for _ in range(50):
    v, g = vg(z)
jax.block_until_ready(g); ms1 = (time.time() - t0) / 50 * 1e3
z4 = {k: jnp.stack([x] * 4) for k, x in z.items()}; vg4 = jax.jit(jax.vmap(jax.value_and_grad(info.potential_fn)))
v, g = vg4(z4); jax.block_until_ready(g); t0 = time.time()
for _ in range(50):
    v, g = vg4(z4)
jax.block_until_ready(g); ms4 = (time.time() - t0) / 50 * 1e3
print(f"ms per gradient: 1 chain {ms1:.2f}; 4 chains vmap {ms4:.2f}", flush=True)

kernel = NUTS(models.m2, target_accept_prob=cfg.target_accept, init_strategy=init_to_median(num_samples=10), max_tree_depth=cfg.max_tree_depth)
mcmc = MCMC(kernel, num_warmup=a.warmup, num_samples=a.samples, num_chains=cfg.num_chains, chain_method=cfg.chain_method, progress_bar=False)
t0 = time.time(); mcmc.warmup(jax.random.PRNGKey(0), d, **kw, extra_fields=("num_steps", "diverging"), collect_warmup=True)
ws = np.asarray(mcmc.get_extra_fields(group_by_chain=True)["num_steps"]); tw = time.time() - t0
print(f"warmup {a.warmup} it: {tw:.0f}s; num_steps per it (chain mean) by quarter: "
      f"{[round(float(q.mean()), 1) for q in np.array_split(ws, 4, axis=1)]}; max {int(ws.max())}", flush=True)
t0 = time.time(); mcmc.run(jax.random.PRNGKey(1), d, **kw, extra_fields=("num_steps", "diverging")); ts = time.time() - t0
ex = mcmc.get_extra_fields(group_by_chain=True); ns = np.asarray(ex["num_steps"])
print(f"sampling {a.samples} it: {ts:.0f}s; num_steps mean per chain {ns.mean(1).round(1).tolist()}; max {int(ns.max())}; "
      f"divergences {int(np.asarray(ex['diverging']).sum())}", flush=True)
per_it = ns.max(0).mean() * ms4 / 1e3
print(f"projection at this tree size: {per_it:.2f} s/it -> {per_it * (cfg.num_warmup + cfg.num_samples) / 3600:.1f} h per fit "
      f"({cfg.num_warmup}+{cfg.num_samples}); at max depth {cfg.max_tree_depth}: "
      f"{(2 ** cfg.max_tree_depth - 1) * ms4 / 1e3 * (cfg.num_warmup + cfg.num_samples) / 3600:.1f} h", flush=True)
