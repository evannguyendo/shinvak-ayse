import json
import argparse
from pathlib import Path
from typing import List, Dict, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, ValidationError, field_validator
import re


TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}$")


class VariantMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beginning_state: Literal["A", "B"]
    end_state: Literal["A", "B"]
    most_recent_state: Literal["A", "B"]
    most_frequent_state: Literal["A", "B"]
    change_exists: bool


class ExampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str = Field(min_length=1)
    control_path: str = Field(min_length=1)
    conflict_path: str = Field(min_length=1)

    conflict_type: str = Field(min_length=1)
    object_name: str = Field(min_length=1)
    attribute_name: str = Field(min_length=1)

    state_A: str = Field(min_length=1)
    state_B: str = Field(min_length=1)

    edit_timestamps: List[str] = Field(default_factory=list)
    change_position_bin: Literal["beginning", "middle", "end"]
    video_duration_sec: float = Field(gt=0)

    control_metadata: VariantMetadata
    conflict_metadata: VariantMetadata

    @field_validator("edit_timestamps")
    @classmethod
    def validate_edit_timestamps(cls, v: List[str]) -> List[str]:
        for item in v:
            if not TIMESTAMP_RE.match(item):
                raise ValueError(f"Invalid edit timestamp '{item}'. Expected MM:SS format.")
        return v


class OptionTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str = Field(min_length=1)
    text_template: str = Field(min_length=1)


class TemplateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str = Field(min_length=1)
    question_family: str = Field(min_length=1)
    applies_to: List[str] = Field(min_length=1)
    question_text_template: str = Field(min_length=1)
    options: List[OptionTemplate] = Field(min_length=1)
    labeling_rule: str = Field(min_length=1)


class TemplateFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    description: str
    templates: List[TemplateRecord]


class CompiledOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str
    text: str


class CompiledQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    template_id: str
    question_family: str
    question_text: str
    options: List[CompiledOption]

    correct_option_control: str
    correct_option_conflict: str

    policy_map_control: Optional[Dict[str, str]] = None
    policy_map_conflict: Optional[Dict[str, str]] = None


class CompiledExampleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    control_path: str
    conflict_path: str

    conflict_type: str
    object_name: str
    attribute_name: str
    state_A: str
    state_B: str

    edit_timestamps: List[str]
    change_position_bin: str
    video_duration_sec: float

    control_metadata: Dict[str, object]
    conflict_metadata: Dict[str, object]

    questions: List[CompiledQuestion]

'''
def load_examples_json(path: str | Path) -> List[ExampleRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Examples file not found: {path}")

    examples: List[ExampleRecord] = []
    seen_video_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)
            example = ExampleRecord.model_validate(raw)

            if example.video_id in seen_video_ids:
                raise RuntimeError(f"Duplicate video_id '{example.video_id}' in {path} line {line_number}")

            seen_video_ids.add(example.video_id)
            examples.append(example)

    return examples
'''

def load_examples_json(path: str | Path) -> List[ExampleRecord]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Examples file not found: {path}")

    seen_video_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Try standard JSON first
    try:
        raw_data = json.loads(text)
        if isinstance(raw_data, list):
            raw_examples = raw_data
        elif isinstance(raw_data, dict):
            raw_examples = [raw_data]
        else:
            raise ValueError("examples file must contain a JSON object or list of objects")
    except json.JSONDecodeError:
        # Fall back to JSONL
        raw_examples = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {e.msg} "
                    f"(column {e.colno})\nLine content: {line[:200]}"
                ) from e

    examples: List[ExampleRecord] = []
    for i, raw in enumerate(raw_examples, start=1):
        example = ExampleRecord.model_validate(raw)

        if example.video_id in seen_video_ids:
            raise RuntimeError(f"Duplicate video_id '{example.video_id}' in {path} record {i}")

        seen_video_ids.add(example.video_id)
        examples.append(example)

    return examples


def load_templates(path: str | Path) -> TemplateFile:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Templates file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return TemplateFile.model_validate(raw)


def render_text(template: str, example: ExampleRecord) -> str:
    return template.format(
        object_name=example.object_name,
        attribute_name=example.attribute_name,
        state_A=example.state_A,
        state_B=example.state_B,
    )


def get_variant_metadata(example: ExampleRecord, variant: Literal["control", "conflict"]) -> VariantMetadata:
    return example.control_metadata if variant == "control" else example.conflict_metadata


def get_correct_option(template: TemplateRecord, example: ExampleRecord, variant: Literal["control", "conflict"]) -> str:
    meta = get_variant_metadata(example, variant)

    if template.labeling_rule == "beginning_state":
        return meta.beginning_state

    if template.labeling_rule == "end_state":
        return meta.end_state

    if template.labeling_rule == "conflict_detection":
        return "A" if meta.change_exists else "B"

    if template.labeling_rule == "change_localization":
        if meta.change_exists:
            return {
                "beginning": "A",
                "middle": "B",
                "end": "C",
            }[example.change_position_bin]
        return "D"

    if template.labeling_rule == "multi_state":
        return "C" if meta.change_exists else "A"

    if template.labeling_rule == "order":
        return "A" if meta.change_exists else "C"

    if template.labeling_rule == "global_state_policy":
        if not meta.change_exists:
            return meta.most_frequent_state
        return "C"

    raise ValueError(f"Unknown labeling_rule: {template.labeling_rule}")


def build_policy_map(template: TemplateRecord, example: ExampleRecord, variant: Literal["control", "conflict"]) -> Optional[Dict[str, str]]:
    if template.labeling_rule != "global_state_policy":
        return None

    meta = get_variant_metadata(example, variant)

    if not meta.change_exists:
        return {
            "A": "single_state",
            "B": "other",
            "C": "multi_state",
            "D": "abstain",
        }

    policy_map: Dict[str, str] = {
        "C": "multi_state",
        "D": "abstain",
    }

    recent = meta.most_recent_state
    frequent = meta.most_frequent_state

    if recent == frequent:
        policy_map["A"] = "ambiguous" if recent == "A" else "other"
        policy_map["B"] = "ambiguous" if recent == "B" else "other"
    else:
        policy_map[recent] = "recency"
        policy_map[frequent] = "frequency"

        other_state = "A" if recent == "B" and frequent != "A" else "B"
        if other_state not in policy_map:
            policy_map[other_state] = "other"

    if "A" not in policy_map:
        policy_map["A"] = "other"
    if "B" not in policy_map:
        policy_map["B"] = "other"

    return policy_map


def compile_example(example: ExampleRecord, templates: List[TemplateRecord]) -> CompiledExampleRecord:
    applicable_templates = [t for t in templates if example.conflict_type in t.applies_to]

    if not applicable_templates:
        raise RuntimeError(f"No templates found for conflict_type='{example.conflict_type}'")

    compiled_questions: List[CompiledQuestion] = []

    for template in applicable_templates:
        question_text = render_text(template.question_text_template, example)
        options = [
            CompiledOption(
                option_id=option.option_id,
                text=render_text(option.text_template, example),
            )
            for option in template.options
        ]

        question_id = f"{template.question_family}"

        compiled_questions.append(
            CompiledQuestion(
                question_id=question_id,
                template_id=template.template_id,
                question_family=template.question_family,
                question_text=question_text,
                options=options,
                correct_option_control=get_correct_option(template, example, "control"),
                correct_option_conflict=get_correct_option(template, example, "conflict"),
                policy_map_control=build_policy_map(template, example, "control"),
                policy_map_conflict=build_policy_map(template, example, "conflict"),
            )
        )

    return CompiledExampleRecord(
        video_id=example.video_id,
        control_path=example.control_path,
        conflict_path=example.conflict_path,
        conflict_type=example.conflict_type,
        object_name=example.object_name,
        attribute_name=example.attribute_name,
        state_A=example.state_A,
        state_B=example.state_B,
        edit_timestamps=example.edit_timestamps,
        change_position_bin=example.change_position_bin,
        video_duration_sec=example.video_duration_sec,
        control_metadata=example.control_metadata.model_dump(),
        conflict_metadata=example.conflict_metadata.model_dump(),
        questions=compiled_questions,
    )


def write_compiled_json(path: str | Path, compiled_examples: List[CompiledExampleRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in compiled_examples:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", required=True, help="Path to metadata-only examples.json")
    parser.add_argument("--templates", required=True, help="Path to question_templates.json")
    parser.add_argument("--output", required=True, help="Path to compiled_examples.json")
    args = parser.parse_args()

    examples = load_examples_json(args.examples)
    template_file = load_templates(args.templates)

    compiled_examples = [compile_example(example, template_file.templates) for example in examples]
    write_compiled_json(args.output, compiled_examples)

    print(f"Loaded {len(examples)} examples")
    print(f"Loaded {len(template_file.templates)} templates")
    print(f"Wrote {len(compiled_examples)} compiled examples to {args.output}")


if __name__ == "__main__":
    main()
