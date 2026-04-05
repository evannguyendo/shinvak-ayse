import os
import cv2
import json
import base64
import zipfile
import tempfile
import shutil
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from dotenv import load_dotenv

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. Load API keys from .env
load_dotenv()
_router_key = os.environ.get("OPENROUTER_API_KEY")
if not _router_key:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. "
        "Create a .env file with OPENROUTER_API_KEY=<your key> and try again."
    )
os.environ["OPENAI_API_KEY"] = _router_key
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# 2. Task → video source mapping (aligned with OpenGVLab mvbench.ipynb)
#    action_sequence / STAR Charades → star.zip
#    action_antonym / SSv2 → ssv2_video.zip
#    action_count / Perception Test → perception.zip  (video_NNNN.mp4)
#    action_localization / STA → sta.zip
TASK_VIDEO_SOURCES = {
    "action_sequence":    {"type": "zip", "file": "video/star.zip"},
    "action_antonym":     {"type": "zip", "file": "video/ssv2_video.zip"},
    "action_count":       {"type": "zip", "file": "video/perception.zip"},
    "action_localization":{"type": "zip", "file": "video/sta.zip"},
}

# 3. Structured output schema
class VideoEvalOutput(BaseModel):
    chain_of_thought: str = Field(description="Briefly explain the sequence of events seen in the frames.")
    predicted_answer: str = Field(description="The exact text of the correct option.")
    confidence_score: int = Field(description="Confidence from 1 to 10")

# 4. Initialize client
client = instructor.from_openai(
    OpenAI(
        default_headers={
            "HTTP-Referer": "https://github.com/evannguyendo",
            "X-Title": "MVBench Eval"
        }
    ),
    mode=instructor.Mode.JSON,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_frames(video_path, num_frames=8):
    """Open a video file, extract evenly-spaced frames, return as Base64 strings."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    actual_frames = min(num_frames, total_frames)
    frame_indices = [int(i * total_frames / actual_frames) for i in range(actual_frames)]
    base64_frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (512, 512))
            ok, buffer = cv2.imencode('.jpg', frame)
            if ok:
                base64_frames.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    return base64_frames


def get_video_from_zip(zip_path, video_name, temp_dir):
    """Extract a single video from a zip archive and return its local path."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith(video_name):
                zip_ref.extract(member, temp_dir)
                return os.path.join(temp_dir, member)
    return None


def get_video_from_ssv2(video_name, temp_dir):
    """
    Download an individual SSv2 .mp4 from HuggingFace (video/ssv2_video_mp4/<name>)
    and return its local path.
    """
    # video_name may already include a sub-path; strip any directory prefix
    bare_name = os.path.basename(video_name)
    hf_path = f"video/ssv2_video_mp4/{bare_name}"
    local_path = hf_hub_download(
        repo_id="OpenGVLab/MVBench",
        repo_type="dataset",
        filename=hf_path,
    )
    return local_path


def evaluate_task(task_name, source_cfg, num_samples=3):
    """
    Load a single MVBench task, resolve its videos, and run model evaluation.
    Returns (correct_predictions, total_processed).
    """
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"{'='*60}")

    dataset = load_dataset("OpenGVLab/MVBench", task_name, split="train")

    temp_dir = tempfile.mkdtemp()
    zip_path = None
    correct_predictions = 0
    total_processed = 0

    try:
        # Pre-download zip archive once for zip-based tasks
        if source_cfg["type"] == "zip":
            print(f"  Downloading/verifying archive: {source_cfg['file']} ...")
            zip_path = hf_hub_download(
                repo_id="OpenGVLab/MVBench",
                repo_type="dataset",
                filename=source_cfg["file"],
            )

        for i in range(min(num_samples, len(dataset))):
            row = dataset[i]
            video_name   = row['video']
            question     = row['question']
            candidates   = row['candidates']
            ground_truth = row['answer']

            print(f"\n  --- Sample {i+1} ---")
            print(f"  Video     : {video_name}")
            print(f"  Question  : {question}")
            print(f"  Truth     : {ground_truth}")

            try:
                if source_cfg["type"] == "zip":
                    local_video_path = get_video_from_zip(zip_path, video_name, temp_dir)
                else:  # ssv2_dir
                    local_video_path = get_video_from_ssv2(video_name, temp_dir)

                if not local_video_path or not os.path.exists(local_video_path):
                    print("  [SKIP] Video not found.")
                    continue

                frames_b64 = extract_frames(local_video_path, num_frames=8)
                if not frames_b64:
                    print("  [SKIP] Could not extract frames.")
                    continue

                options_text = "\n".join([f"- {opt}" for opt in candidates])
                full_prompt  = f"{question}\n\nChoose the exact correct answer from these options:\n{options_text}"

                content = [{"type": "text", "text": full_prompt}]
                for frame in frames_b64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                    })

                print("  Querying OpenRouter (GPT-4o)...")
                eval_result = client.chat.completions.create(
                    model="openai/gpt-4o",
                    response_model=VideoEvalOutput,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert video analysis AI. Review the frames in chronological order.",
                        },
                        {"role": "user", "content": content},
                    ],
                )

                is_correct = (
                    eval_result.predicted_answer.strip().lower()
                    == ground_truth.strip().lower()
                )
                if is_correct:
                    correct_predictions += 1
                total_processed += 1

                print(f"  Reasoning : {eval_result.chain_of_thought}")
                print(f"  Prediction: {eval_result.predicted_answer}")
                print(f"  Result    : {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

            except Exception as e:
                print(f"  [ERROR] {e}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return correct_predictions, total_processed


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    results = {}

    for task_name, source_cfg in TASK_VIDEO_SOURCES.items():
        try:
            correct, total = evaluate_task(task_name, source_cfg, num_samples=3)
            results[task_name] = (correct, total)
        except Exception as e:
            print(f"\n  [TASK ERROR] {task_name}: {e}")
            results[task_name] = (0, 0)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    overall_correct = overall_total = 0
    for task_name, (correct, total) in results.items():
        if total > 0:
            acc = correct / total * 100
            print(f"  {task_name:<25} {acc:5.1f}%  ({correct}/{total})")
        else:
            print(f"  {task_name:<25}  N/A   (0 samples processed)")
        overall_correct += correct
        overall_total   += total

    if overall_total > 0:
        overall_acc = overall_correct / overall_total * 100
        print(f"\n  {'OVERALL':<25} {overall_acc:5.1f}%  ({overall_correct}/{overall_total})")


if __name__ == "__main__":
    main()
