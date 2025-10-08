"""Command line script for supervised fine-tuning Qwen2.5-Omni models.

This script uses the Hugging Face `transformers` training stack to fine-tune a
`Qwen2.5-Omni` checkpoint on conversation style supervised data.  The input
dataset is expected to contain one conversation per row under a field named
``messages`` whose value is a list of dictionaries compatible with the
tokenizer chat template (i.e. each item should have a ``role`` field such as
``"user"`` or ``"assistant"`` and a ``content`` field holding the text).

Example JSONL row::

    {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下Qwen2.5-Omni。"},
        {"role": "assistant", "content": "Qwen2.5-Omni 是..."}
    ]}

The script accepts either a local JSON/JSONL file or a dataset hub identifier.
When an evaluation dataset is supplied the same preprocessing pipeline is used
and the perplexity is reported at the end of training.

Typical invocation::

    python cookbooks/train_qwen2_5_omni_sft.py \
        --model-name Qwen/Qwen2.5-Omni \
        --dataset-path /path/to/sft_data.jsonl \
        --output-dir ./qwen2_5_omni_sft \
        --per-device-train-batch-size 1 \
        --gradient-accumulation-steps 16 \
        --num-train-epochs 3

For large models make sure to install the dependencies listed in
``requirements_web_demo.txt`` and consider enabling gradient checkpointing or
QLoRA/PEFT depending on the available hardware.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def _load_json_file(path: str) -> Dataset:
    """Load a JSON/JSONL file with conversation data into a Dataset object."""

    data_files: Dict[str, str] = {"train": path}
    extension = os.path.splitext(path)[1]
    if extension not in {".json", ".jsonl"}:
        raise ValueError(
            f"Unsupported dataset file extension '{extension}'. "
            "Please provide a JSON or JSONL file."
        )
    return load_dataset("json", data_files=data_files)["train"]


def load_conversation_dataset(dataset_path: str) -> Dataset:
    """Load the training dataset from either a file path or a hub identifier."""

    if os.path.isfile(dataset_path):
        return _load_json_file(dataset_path)

    dataset = load_dataset(dataset_path)
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(
                "The provided dataset does not contain a 'train' split."
            )
        return dataset["train"]
    return dataset  # type: ignore[return-value]


def load_eval_dataset(eval_path: str) -> Dataset:
    """Load an evaluation dataset, mirroring ``load_conversation_dataset``."""

    if os.path.isfile(eval_path):
        dataset = _load_json_file(eval_path)
        return dataset

    dataset = load_dataset(eval_path)
    if isinstance(dataset, DatasetDict):
        if "validation" in dataset:
            return dataset["validation"]
        if "eval" in dataset:
            return dataset["eval"]
        if "test" in dataset:
            return dataset["test"]
        raise ValueError(
            "Evaluation dataset must contain a 'validation', 'eval', or 'test' split."
        )
    return dataset  # type: ignore[return-value]


def apply_chat_template(
    tokenizer: AutoTokenizer,
    example: Mapping[str, Any],
    messages_field: str,
    default_system: Optional[str],
) -> str:
    """Convert a conversation example into a single training prompt string."""

    messages = list(example[messages_field])  # type: ignore[arg-type]
    if default_system and not any(msg.get("role") == "system" for msg in messages):
        messages = [{"role": "system", "content": default_system}] + messages

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prompt


@dataclass
class ConversationPreprocessor:
    """Callable that transforms raw conversation data into token IDs."""

    tokenizer: AutoTokenizer
    messages_field: str
    default_system: Optional[str]
    max_length: int
    packing: bool

    def __call__(self, examples: Mapping[str, List[Any]]) -> Dict[str, Any]:
        prompts = [
            apply_chat_template(
                self.tokenizer,
                example,
                messages_field=self.messages_field,
                default_system=self.default_system,
            )
            for example in _iterate_examples(examples)
        ]
        tokenized = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]

        if self.packing:
            tokenized = _pack_sequences(tokenized, self.max_length, self.tokenizer.pad_token_id)
        return tokenized


def _iterate_examples(examples: Mapping[str, List[Any]]) -> Iterable[Mapping[str, Any]]:
    keys = list(examples.keys())
    length = len(examples[keys[0]])
    for i in range(length):
        yield {key: examples[key][i] for key in keys}


def _pack_sequences(
    tokenized: Dict[str, List[List[int]]],
    max_length: int,
    pad_token_id: int,
) -> Dict[str, List[List[int]]]:
    """Pack token sequences together to utilise context length more efficiently."""

    packed_inputs: List[List[int]] = []
    packed_attn: List[List[int]] = []
    packed_labels: List[List[int]] = []

    buffer_inputs: List[int] = []
    buffer_labels: List[int] = []

    for input_ids, labels in zip(tokenized["input_ids"], tokenized["labels"]):
        if len(buffer_inputs) + len(input_ids) > max_length:
            if buffer_inputs:
                _finalise_sequence(
                    buffer_inputs,
                    buffer_labels,
                    max_length,
                    pad_token_id,
                    packed_inputs,
                    packed_attn,
                    packed_labels,
                )
                buffer_inputs, buffer_labels = [], []

        buffer_inputs.extend(input_ids)
        buffer_labels.extend(labels)

    if buffer_inputs:
        _finalise_sequence(
            buffer_inputs,
            buffer_labels,
            max_length,
            pad_token_id,
            packed_inputs,
            packed_attn,
            packed_labels,
        )

    return {
        "input_ids": packed_inputs,
        "attention_mask": packed_attn,
        "labels": packed_labels,
    }


def _finalise_sequence(
    buffer_inputs: List[int],
    buffer_labels: List[int],
    max_length: int,
    pad_token_id: int,
    packed_inputs: List[List[int]],
    packed_attn: List[List[int]],
    packed_labels: List[List[int]],
) -> None:
    buffer_inputs = buffer_inputs[:max_length]
    buffer_labels = buffer_labels[:max_length]
    attention_mask = [1] * len(buffer_inputs)

    padding = max_length - len(buffer_inputs)
    if padding > 0:
        buffer_inputs = buffer_inputs + [pad_token_id] * padding
        buffer_labels = buffer_labels + [-100] * padding
        attention_mask = attention_mask + [0] * padding

    packed_inputs.append(buffer_inputs)
    packed_attn.append(attention_mask)
    packed_labels.append(buffer_labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-Omni",
        help="Model identifier from the Hugging Face hub or local path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path or hub identifier for the training dataset.",
    )
    parser.add_argument(
        "--eval-dataset-path",
        type=str,
        default=None,
        help="Optional evaluation dataset path or hub identifier.",
    )
    parser.add_argument(
        "--messages-field",
        type=str,
        default="messages",
        help="Field name in the dataset that holds the conversation turns.",
    )
    parser.add_argument(
        "--default-system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend when a conversation has no system role.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the fine-tuned model and checkpoints will be stored.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for the AdamW optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for the learning rate scheduler.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every X update steps.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log training metrics every X update steps.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for lower memory usage.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Train using bfloat16 precision when supported.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Enable TF32 on NVIDIA Ampere+ GPUs for matmul operations.",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Concatenate conversations to better utilise the context window.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialisation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_conversation_dataset(args.dataset_path)

    preprocessor = ConversationPreprocessor(
        tokenizer=tokenizer,
        messages_field=args.messages_field,
        default_system=args.default_system_prompt,
        max_length=args.max_length,
        packing=args.packing,
    )
    train_dataset = train_dataset.map(
        preprocessor,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_eval_dataset(args.eval_dataset_path)
        eval_dataset = eval_dataset.map(
            preprocessor,
            batched=True,
            remove_columns=eval_dataset.column_names,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps" if eval_dataset is not None else "no",
        save_total_limit=5,
        bf16=args.bf16,
        report_to="none",
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
