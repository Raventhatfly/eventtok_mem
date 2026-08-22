"""Head-to-head: the learned tokenizer against k-means, at a matched codebook size.

    python -m eventtok.scripts.report_neural_vs_kmeans

The project has run entirely on k-means, and the claim that a learned tokenizer would
have to beat Lloyd's algorithm was an assertion, never a measurement. This pairs each
task's neural row with the k-means row at the same codebook size and reports the
difference on the metrics that matter downstream.

Matching the codebook size is the whole point of the comparison. The k sweep already
established that label accuracy climbs with the number of clusters more or less
regardless of anything else, so a learned tokenizer with a larger vocabulary would win
for a reason that has nothing to do with learning.

Two metrics are reported because they can disagree, and did for k-means:
``token_gain`` is the identity of the BPE tokens that would enter the log, and
``bpe_f1`` is whether their boundaries sit where events actually change.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .. import paths


def load(pattern):
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            r = json.load(fh)
        out[r["task"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64, help="k-means codebook to match")
    ap.add_argument("--variant", default="neural-av", choices=["neural-a", "neural-av"])
    args = ap.parse_args()

    E = str(paths.CACHE_ROOT / "eval")
    km = load(os.path.join(E, f"alltasks_k{args.k}_*.json"))
    nn = load(os.path.join(E, f"neural_*_{args.variant}.json"))
    shared = sorted(set(km) & set(nn))
    if not shared:
        raise SystemExit(
            f"no overlap between k-means k={args.k} ({len(km)} tasks) and "
            f"{args.variant} ({len(nn)} tasks)"
        )

    w = max(len(t) for t in shared)
    print(f"\n  {args.variant} vs k-means k={args.k}, {len(shared)} tasks\n")
    print(f"  {'task':<{w}} {'kmeans tok':>11} {'neural tok':>11} {'Δ':>7} │ "
          f"{'km F1':>7} {'nn F1':>7} {'Δ':>7} │ {'live':>9}")
    print("  " + "─" * (w + 66))
    dt, df = [], []
    for t in shared:
        a, b = km[t], nn[t]
        d_tok = b["token_gain"] - a["token_gain"]
        d_f1 = b["bpe_f1"] - a["bpe_f1"]
        dt.append(d_tok)
        df.append(d_f1)
        live = f"{b.get('live_codes', 0)}/{b.get('codebook_size', 0)}"
        print(
            f"  {t:<{w}} {a['token_gain']:>+10.1%} {b['token_gain']:>+10.1%} "
            f"{d_tok:>+7.1%} │ {a['bpe_f1']:>7.3f} {b['bpe_f1']:>7.3f} "
            f"{d_f1:>+7.3f} │ {live:>9}"
        )

    dt, df = np.array(dt), np.array(df)
    print(
        f"\n  token identity gain: neural minus k-means, median {np.median(dt):+.1%}, "
        f"neural ahead on {int((dt > 0).sum())}/{len(dt)}"
    )
    print(
        f"  boundary F1:         neural minus k-means, median {np.median(df):+.3f}, "
        f"neural ahead on {int((df > 0).sum())}/{len(df)}"
    )
    live = np.array([nn[t].get("live_codes", 0) for t in shared])
    cap = np.array([nn[t].get("codebook_size", 1) for t in shared])
    print(
        f"  codebook use: neural keeps {np.median(live / cap):.0%} of its codes alive "
        f"(median), k-means uses all {args.k} by construction"
    )
    if np.median(dt) <= 0 and np.median(df) <= 0:
        print(
            "\n  The learned tokenizer does not beat Lloyd's algorithm on either metric.\n"
            "  On this evidence the encoder, the FSQ bottleneck and the training loop\n"
            "  are not paying for themselves and the k-means path is the honest baseline\n"
            "  to build on."
        )
    elif np.median(dt) > 0 and np.median(df) > 0:
        print("\n  The learned tokenizer wins on both metrics.")
    else:
        print("\n  Split decision: the two metrics disagree, so neither is the winner.")


if __name__ == "__main__":
    main()
