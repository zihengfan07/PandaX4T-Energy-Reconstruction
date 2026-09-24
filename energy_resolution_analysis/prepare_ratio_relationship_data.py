#!/usr/bin/env python3
"""Prepare compact JSON data for ratio-versus-variable plots."""

import argparse
import json
import numpy as np
import pandas as pd

VARIABLES = [
    ("dt", "Drift time", 0.001, "us"),
    ("wS2CDF_max", "S2 CDF width", 1.0, "raw unit"),
    ("yS2Tcor_max", "Corrected top y", 1.0, "mm"),
    ("xS2Bcor_max", "Corrected bottom x", 1.0, "mm"),
]

def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--energy", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.raw, sep="\t", usecols=[v[0] for v in VARIABLES], low_memory=False)
    energy = pd.read_csv(args.energy, usecols=["energy_cor_kev", args.after], low_memory=False)
    if len(raw) != len(energy): raise ValueError("row mismatch")
    ratio = energy[args.after].to_numpy(float) / energy.energy_cor_kev.to_numpy(float)
    valid_ratio = np.isfinite(ratio) & (ratio > 0)
    rng = np.random.RandomState(20260815)
    base_idx = np.flatnonzero(valid_ratio)
    sample = np.sort(rng.choice(base_idx, size=min(args.sample, len(base_idx)), replace=False))
    panels = []
    for column, label, scale, unit in VARIABLES:
        x = raw[column].to_numpy(float) * scale
        valid = valid_ratio & np.isfinite(x)
        lo, hi = np.quantile(x[valid], [0.005, 0.995])
        show = valid & (x >= lo) & (x <= hi)
        edges = np.unique(np.quantile(x[show], np.linspace(0, 1, 19)))
        bins = []
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            selected = show & (x >= left) & (x <= right if index == len(edges)-2 else x < right)
            if np.any(selected):
                q16, q50, q84 = np.quantile(ratio[selected], [0.16, 0.50, 0.84])
                bins.append([round(float(np.median(x[selected])),5), round(float(q16),7),
                             round(float(q50),7), round(float(q84),7), int(np.sum(selected))])
        si = sample[np.isfinite(x[sample]) & (x[sample] >= lo) & (x[sample] <= hi)]
        points = [[round(float(x[i]),5), round(float(ratio[i]),7), int(abs(ratio[i]-1)>1e-12)] for i in si]
        applied = valid & (np.abs(ratio-1)>1e-12)
        panels.append({"key":column,"label":label,"unit":unit,
                       "rho":round(spearman(x[valid],ratio[valid]),3),
                       "rhoApplied":round(spearman(x[applied],ratio[applied]),3),
                       "xDomain":[float(lo),float(hi)],"points":points,"bins":bins})
    payload = {"panels":panels,
               "ratioDomain":[float(np.quantile(ratio[valid_ratio],.003)),float(np.quantile(ratio[valid_ratio],.997))],
               "n":int(np.sum(valid_ratio)),"applied":int(np.sum(valid_ratio & (np.abs(ratio-1)>1e-12)))}
    with open(args.output,"w") as handle: json.dump(payload,handle,separators=(",",":"))
    print(args.output)

if __name__ == "__main__": main()
