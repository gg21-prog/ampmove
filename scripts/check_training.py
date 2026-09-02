#!/usr/bin/env python3
"""
Training monitor for AMP run.
Reads TensorBoard events + log file and prints a go/no-go assessment.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG  = REPO / "logs/run1.log"
TB   = REPO / "logs/amp_ironcub/run1"

# ── Step from tqdm progress bar ────────────────────────────────────────────────
def get_step():
    if not LOG.exists():
        return 0
    # tqdm lines look like: | 12345/1000000 [...]
    step = 0
    with open(LOG, "rb") as f:
        # Read last 8KB for speed
        f.seek(max(0, f.seek(0, 2) - 8192 or 0))
        chunk = f.read().decode("utf-8", errors="ignore")
    for m in re.finditer(r"\|\s*(\d+)/1000000", chunk):
        step = max(step, int(m.group(1)))
    return step

# ── Metrics from TensorBoard ───────────────────────────────────────────────────
def get_metrics():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    if not TB.exists():
        return {}
    ea = EventAccumulator(str(TB), size_guidance={"scalars": 200})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    metrics = {}
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            metrics[tag] = events[-1].value
    return metrics

# ── Assessment ─────────────────────────────────────────────────────────────────
def assess(step, metrics):
    lines = []
    verdict = "UNKNOWN"

    ep_len  = metrics.get("Episode / Total timesteps (mean)")
    ep_max  = metrics.get("Episode / Total timesteps (max)")
    disc_l  = metrics.get("Loss / Discriminator loss")
    rew_m   = metrics.get("Reward / Instantaneous reward (mean)")
    tot_rew = metrics.get("Reward / Total reward (mean)")
    style_r = None   # SKRL logs style inside instantaneous reward; use disc_l as proxy
    task_r  = rew_m

    lines.append(f"Step: {step:,} / 1,000,000  ({step/1e6*100:.1f}%)")
    lines.append(f"  ep_length (mean/max) : {ep_len:.1f} / {ep_max:.0f}" if ep_len is not None else "  ep_length      : (not yet logged)")
    lines.append(f"  disc_loss      : {disc_l:.4f}  (healthy=0.3–0.7, collapsed=<0.1)" if disc_l is not None else "  disc_loss      : (not yet logged)")
    lines.append(f"  reward/step    : {rew_m:.3f}"  if rew_m is not None else "  reward/step    : (not yet logged)")
    lines.append(f"  total_reward   : {tot_rew:.1f}" if tot_rew is not None else "  total_reward   : (not yet logged)")

    # --- Go/no-go logic ---
    if step < 5_000:
        verdict = "TOO EARLY — wait for 10K+"
    elif step < 10_000:
        if ep_len is not None and ep_len < 5:
            verdict = "⚠️  WARNING: episode length < 5 at 10K — likely crashing on reset"
        else:
            verdict = "EARLY — check again at 50K"
    elif step < 60_000:
        # 10K–60K sanity check
        bad = 0
        if ep_len  is not None and ep_len  < 10:  bad += 1; lines.append("  ⚠️  episode_length very short")
        if style_r is not None and style_r < 0.01: bad += 1; lines.append("  ⚠️  style_reward near zero")
        if disc_l  is not None and disc_l  < 0.1:  bad += 1; lines.append("  ⚠️  discriminator collapsed")
        verdict = "⚠️  CONCERN — consider stopping" if bad >= 2 else "OK — keep going, check at 50K"
    else:
        # 50K+ real go/no-go
        green = 0
        red   = 0
        if ep_len is not None:
            if ep_len > 100:  green += 1; lines.append("  ✅ ep_length > 100 (robot staying up)")
            elif ep_len < 30: red   += 1; lines.append("  ❌ ep_length < 30 — robot keeps falling")
        if disc_l is not None:
            if 0.3 < disc_l < 0.8:  green += 1; lines.append("  ✅ discriminator balanced (0.3–0.7)")
            elif disc_l < 0.1:       red   += 1; lines.append("  ❌ discriminator collapsed — lower disc_loss_scale or gp_scale")
            elif disc_l > 0.9:       red   += 1; lines.append("  ❌ discriminator stuck — policy not learning style")
        if tot_rew is not None:
            if tot_rew > 600:  green += 1; lines.append("  ✅ total_reward growing (robot alive + moving)")
            elif tot_rew < 100: red  += 1; lines.append("  ❌ total_reward very low — policy not improving")

        if red >= 2:
            verdict = "🔴 STOP + RETUNE — weights are bad"
        elif green >= 2:
            verdict = "🟢 LOOKING GOOD — let it run"
        else:
            verdict = "🟡 MIXED — check W&B curves for trend"

    lines.append(f"\nVerdict: {verdict}")
    return "\n".join(lines)


if __name__ == "__main__":
    step    = get_step()
    try:
        metrics = get_metrics()
    except Exception as e:
        metrics = {}
        print(f"(metrics unavailable: {e})", file=sys.stderr)
    print(assess(step, metrics))
