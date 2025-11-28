import argparse
import subprocess
import os
import itertools


def run(cmd: list):
    print("\n=== Running: " + " ".join(str(c) for c in cmd) + "\n")
    subprocess.run(cmd, check=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def run_sampling(model, model_id, bench, device, dtype, max_tokens, num_trials, temp, top_p, top_k):
    script = "evaluation/inference_sampling.py"
    run_name = f"sampling_t{temp}_p{top_p}_k{top_k}"
    cmd = [
        "python", script,
        "--model-path", model,
        "--model-id", model_id,
        "--bench-name", bench,
        "--device", device,
        "--dtype", dtype,
        "--max-tokens", str(max_tokens),
        "--num-trials", str(num_trials),
        "--temperature", str(temp),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--run-name", run_name
    ]
    run(cmd)


def run_speculative(
    model,
    assistant,
    model_id,
    bench,
    device,
    dtype,
    max_tokens,
    num_trials,
    num_assist,
    conf_thresh,
    schedule,
    temp,
    top_p,
    top_k,
):
    script = "evaluation/inference_speculative.py"
    run_name = (
        f"spec_nt{num_assist}_th{conf_thresh}_{schedule}_"
        f"t{temp}_p{top_p}_k{top_k}"
    )

    cmd = [
        "python", script,
        "--model-path", model,
        "--assistant-model-path", assistant,
        "--model-id", model_id,
        "--bench-name", bench,
        "--device", device,
        "--dtype", dtype,
        "--max-tokens", str(max_tokens),
        "--num-trials", str(num_trials),
        "--num-assistant-tokens", str(num_assist),
        "--assistant-confidence-threshold", str(conf_thresh),
        "--num-assistant-tokens-schedule", schedule,
        "--temperature", str(temp),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--run-name", run_name
    ]
    run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--bench", type=str, default="gsm8k")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--num-trials", type=int, default=3)

    parser.add_argument("--models", nargs="+", required=True,
                        help="List of main models (e.g., google/gemma-2-2b-it)")
    parser.add_argument("--assistants", nargs="+", default=[],
                        help="List of assistant models (same length or broadcast 1-to-many)")

    args = parser.parse_args()

    if len(args.assistants) == 1 and len(args.models) > 1:
        args.assistants = args.assistants * len(args.models)

    assert len(args.assistants) in (0, len(args.models)), \
        "assistants must be empty or match #models or broadcast to it."

    temperatures = [0.7, 1.0]
    top_ps = [0.9, 0.95]
    top_ks = [20, 50]

    speculative_nt = [5, 10, 20]
    speculative_th = [0.3, 0.4]
    speculative_schedules = ["constant", "heuristic"]

    ensure_dir("experiments")

    for idx, model in enumerate(args.models):

        model_id = model.split("/")[-1]
        assistant = args.assistants[idx] if args.assistants else None

        print(f"\n==============================")
        print(f" Running model: {model_id}")
        print(f"==============================")

        # ----------------------------
        # 1. Sampling experiments
        # ----------------------------
        for temp, p, k in itertools.product(temperatures, top_ps, top_ks):
            run_sampling(
                model=model,
                model_id=model_id,
                bench=args.bench,
                device=args.device,
                dtype=args.dtype,
                max_tokens=args.max_tokens,
                num_trials=args.num_trials,
                temp=temp,
                top_p=p,
                top_k=k
            )

        # ----------------------------
        # 2. Speculative decoding (if assistant provided)
        # ----------------------------
        if assistant:
            for nt, th, sched, temp, p, k in itertools.product(
                speculative_nt,
                speculative_th,
                speculative_schedules,
                temperatures,
                top_ps,
                top_ks
            ):
                run_speculative(
                    model=model,
                    assistant=assistant,
                    model_id=model_id,
                    bench=args.bench,
                    device=args.device,
                    dtype=args.dtype,
                    max_tokens=args.max_tokens,
                    num_trials=args.num_trials,
                    num_assist=nt,
                    conf_thresh=th,
                    schedule=sched,
                    temp=temp,
                    top_p=p,
                    top_k=k
                )


## Commands

# Sampling only (no speculative):
# python run_experiments.py \
#     --models google/gemma-2-2b-it \
#     --device cpu

# Sampling + Speculative:
# python run_experiments.py \
#     --models google/gemma-2-4b-it \
#     --assistants google/gemma-2-2b-it \
#     --device cpu

# Multiple models:
# python run_experiments.py \
#     --models google/gemma-2-1b-it google/gemma-2-4b-it \
#     --assistants google/gemma-2-1b-it google/gemma-2-2b-it \
#     --device cpu

