# Frame-count (edit-magnitude) ablation

Varies **how much of the edited segment is spliced in**, to measure the dose-response of when
video-LLMs detect a temporal conflict. Five variants per sample:

| variant | edited frames |
|---|---|
| `splice_1frame` | 1 (minimal edit) |
| `splice_10` | first 10% of the edited segment |
| `splice_20` | 20% |
| `splice_50` | 50% |
| `splice_100` | 100% (≡ existing `final_spliced.mp4`) |

Built by `frame_ablation.py` from clips already in `hf_conflict_outputs/` — **pure frame
manipulation, no AI generation / no API credits.** Method: `p0000a_orig + REGION + ptail_orig`
where `REGION[i]` is the edited frame for `i < n` else the original frame
(`source_clips/seg_000_src.mp4`); `n = round(fraction * segment_frames)`, `n=1` for 1-frame.
Anchor = start of segment. Output length matches `final_spliced` (100% reproduces it exactly).

Coverage: **72 / 75 samples** (the 3 skipped `object_interaction` samples have no edited segment —
same 3 already absent from the edited inference set, where `edited_total = 72`).

## Regenerate (the .mp4s are git-ignored except one preview sample)
```bash
python frame_ablation.py          # writes 360 videos to ablation_outputs/<id>/  (~2.5 min, no API)
python build_ablation_examples.py # writes ablation_examples.jsonl  (360 records, inference-ready)
```

## Files committed
- `ablation_manifest.json` — index over all samples (frames, fractions, paths).
- `<id>/ablation_meta.json` — per-sample metadata.
- `ablation_examples.jsonl` — 360 inference records (one per sample×variant); `conflict_path` is the
  spliced video, `control_path` the original. Carries `fraction`, `edited_frames`, question + options.
- `moving_attribute_00/splice_*.mp4` — one example sample kept as a visual preview.

## Next: run the matrix
Feed `ablation_examples.jsonl` through the inference suite per variant and plot
**conflict-detection rate vs. fraction edited**, by model and by split.
