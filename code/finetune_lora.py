"""
finetune_lora.py -- QLoRA fine-tune of the slot-extraction model, as a plain
script on the GPU pod (no Modal wrappers).

Data: JSONL under data/train/, one object per line:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}],
     "response": "<gold assistant output, e.g. the extraction JSON>"}

Run (on the pod, repo root):
    python code/generate_sft_set.py          # writes data/train/sft_v1.jsonl
    python code/finetune_lora.py             # trains Llama-3.1-8B-Instruct by default
    python code/finetune_lora.py --epochs 2  # override any hyperparameter

After it finishes, serve the adapter alongside the base model:
    LORA_DIR=adapters/triage-lora bash bootstrap.sh
then the eval harness compares model="<base>" vs model="triage-lora".
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # Default is the model actually served at runtime (llm_client.py) and by
    # bootstrap.sh. Keep these three in sync -- a train/serve model mismatch
    # produces an adapter that silently does nothing useful.
    p.add_argument("--model", default=os.environ.get(
        "MODEL", "meta-llama/Llama-3.1-8B-Instruct"))
    p.add_argument("--data", default="data/train", help="dir with *.jsonl")
    p.add_argument("--out", default="adapters/triage-lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch", type=int, default=2, help="per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    # EXTRACT_SYS alone is ~1,640 tokens; add the user turn + gold JSON and a
    # real extraction example is ~1,800-2,000. 1024 (the old default) silently
    # truncated every extraction row's label off the end of the sequence.
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--no-4bit", action="store_true", help="LoRA on fp16/bf16 instead of QLoRA")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    files = sorted(glob.glob(os.path.join(args.data, "*.jsonl")))
    if not files:
        sys.exit(
            f"no *.jsonl in {args.data}/ -- generate the SFT set first "
            f"(code/generate_sft_set.py)."
        )
    print(f"train files: {files}")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"compute dtype: {compute_dtype}, 4-bit: {not args.no_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if not args.no_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_cfg,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )

    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # -- build a single "text" column from messages + gold response ----------
    def to_text(ex: dict) -> dict:
        convo = ex["messages"] + [{"role": "assistant", "content": ex["response"]}]
        return {"text": tokenizer.apply_chat_template(convo, tokenize=False)}

    ds = load_dataset("json", data_files=files, split="train")
    ds = ds.map(to_text, remove_columns=list(ds.column_names))
    print(f"examples: {len(ds)}")
    print("--- sample formatted example ---")
    print(ds[0]["text"][:800])
    print("--- end sample ---")

    # Guard against the old silent-truncation bug: if a meaningful fraction of
    # rows tokenize longer than max_seq_len, the gold response gets cut off and
    # the model trains on nothing useful for those rows.
    lens = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in ds["text"]]
    over = sum(1 for n in lens if n > args.max_seq_len)
    print(f"token lengths: max={max(lens)} p95={sorted(lens)[int(0.95 * (len(lens) - 1))]} "
          f"mean={sum(lens) / len(lens):.0f}  |  over max_seq_len({args.max_seq_len}): {over}/{len(lens)}")
    if over > 0.02 * len(lens):
        print(f"WARNING: {over} rows exceed --max-seq-len {args.max_seq_len}; "
              f"raise it (their labels are being truncated).")

    # -- trl API moved SFT args into SFTConfig around 0.11; support both -----
    from trl import SFTTrainer

    common = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        bf16=bf16_ok,
        fp16=not bf16_ok,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    )
    try:
        from trl import SFTConfig

        sft_args = SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            packing=False,
            **common,
        )
        trainer = SFTTrainer(model=model, args=sft_args, train_dataset=ds)
    except ImportError:
        from transformers import TrainingArguments

        trainer = SFTTrainer(
            model=model,
            args=TrainingArguments(**common),
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            packing=False,
        )

    train_out = trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)

    print("\n==============================================================")
    print(f" adapter saved -> {args.out}")
    print(f" final train loss: {train_out.training_loss:.4f}")
    print(" serve it:  LORA_DIR=%s bash bootstrap.sh" % args.out)
    print(" then eval:  model='%s' (baseline)  vs  model='triage-lora'" % args.model)
    print("==============================================================")


if __name__ == "__main__":
    main()
