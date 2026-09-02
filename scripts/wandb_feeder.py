#!/usr/bin/env python3
"""
Tails the SKRL TensorBoard events file and pushes metrics into the
existing W&B run. Run this in a separate terminal alongside training.

Usage:
    WANDB_API_KEY=<key> python scripts/wandb_feeder.py \
        --run_id el8as9jg --project ampmove-ironcub --entity sarayusapa-vjti
"""
import argparse
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run_id",  required=True, help="W&B run ID to resume")
parser.add_argument("--project", default="ampmove-ironcub")
parser.add_argument("--entity",  default="sarayusapa-vjti")
parser.add_argument("--tb_dir",  default="logs/amp_ironcub/run1",
                    help="Directory containing TF events file")
parser.add_argument("--poll",    type=int, default=30,
                    help="Seconds between polling for new events")
args = parser.parse_args()

TB_DIR = Path(args.tb_dir)

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

print(f"Resuming W&B run {args.run_id} ...")
run = wandb.init(
    project=args.project,
    entity=args.entity,
    id=args.run_id,
    resume="must",
    sync_tensorboard=False,
)
print(f"Attached: {run.url}")

last_logged_step = -1

def push_events():
    global last_logged_step
    ea = EventAccumulator(str(TB_DIR), size_guidance={"scalars": 100_000})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print("  (no scalar tags yet)")
        return

    # Collect all (step -> {tag: value}) mappings
    step_data: dict[int, dict] = {}
    for tag in tags:
        for event in ea.Scalars(tag):
            if event.step > last_logged_step:
                step_data.setdefault(event.step, {})[tag] = event.value

    if not step_data:
        print(f"  no new steps above {last_logged_step}")
        return

    # Log in step order
    for step in sorted(step_data.keys()):
        wandb.log(step_data[step], step=step)

    new_last = max(step_data.keys())
    print(f"  pushed steps {last_logged_step+1}–{new_last}  "
          f"({len(step_data)} steps, {len(tags)} tags)")
    last_logged_step = new_last

print(f"Polling {TB_DIR} every {args.poll}s  (Ctrl+C to stop)\n")
while True:
    try:
        push_events()
    except Exception as e:
        print(f"  error: {e}")
    time.sleep(args.poll)
