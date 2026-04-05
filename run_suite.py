import argparse
import json
import subprocess
from pathlib import Path


def load_models(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("models.json must contain a top-level 'models' list")

    enabled_models = [m for m in models if m.get("enabled", True)]
    if not enabled_models:
        raise RuntimeError("No enabled models found in models config")

    return enabled_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", required=True, help="Path to models.json")
    parser.add_argument("--examples", required=True, help="Path to examples.jsonl")
    parser.add_argument("--out-dir", required=True, help="Directory for run outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit")
    args = parser.parse_args()

    models = load_models(args.models_config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(models)} enabled models")

    for model_cfg in models:
        model_id = model_cfg["id"]
        model_name = model_cfg["openrouter_model"]
        max_tokens = int(model_cfg.get("max_tokens", 1200))
        temperature = float(model_cfg.get("temperature", 0.1))

        model_dir = out_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        responses_path = model_dir / "responses.jsonl"

        cmd = [
            "python",
            "run_batch.py",
            "--examples",
            args.examples,
            "--responses",
            str(responses_path),
            "--model",
            model_name,
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
        ]

        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])

        print(f"\n=== Running {model_id} ({model_name}) ===")
        print(" ".join(cmd))

        completed = subprocess.run(cmd)

        if completed.returncode != 0:
            print(f"[failed] model={model_id} exit_code={completed.returncode}")
        else:
            print(f"[done] model={model_id} responses={responses_path}")


if __name__ == "__main__":
    main()

'''
python run_suite.py \
  --models-config models.json \
  --examples examples.jsonl \
  --out-dir runs/test_run \
  --limit 3
'''