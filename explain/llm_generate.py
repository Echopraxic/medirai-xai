"""
Generate explanations with an open-source LLM (plan WS4/WS5), verify them, and log everything needed to
reproduce the run (model id + revision, prompt version + hash, decoding params, per-item input hash).

Evaluation runs are fully deterministic (greedy decoding, fixed seed), per "Reproducible Prompt Testing".
Only the number-free, de-identified JSON from llm_input.build goes to the model: no image, no metadata,
no identifier (the isic_id stays outside the prompt).

    # on Rorqual (see slurm/rorqual_llm.sh); offline, weights pre-downloaded to $HF_HOME
    python explain/llm_generate.py --inputs trees/output_v0.2/explanation_inputs.jsonl \
        --model mistralai/Mistral-7B-Instruct-v0.3 --prompt cite_v1 --backend vllm --out explain/output
    # no-LLM template arm on the same inputs (also a CPU smoke test of the whole pipeline)
    python explain/llm_generate.py --inputs ... --backend template --out explain/output
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_input as li  # noqa: E402
import verifier  # noqa: E402
from template import render_llm_input  # noqa: E402
import attribution_encodings as encodings  # noqa: E402

DECODING = {"temperature": 0.0, "do_sample": False, "max_new_tokens": 700, "seed": 0}


def load_prompt(version):
    path = HERE / "prompts" / f"{version}.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["sha256_16"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return spec


def messages(spec, inp, prompt_input=None):
    """prompt_input: what the LLM sees when an attribution encoding is set (explain/attribution_encodings.py); else inp."""
    shown = prompt_input or inp
    system = spec["system"].format(audience=shown["audience"])
    user = spec["user"].format(input_json=json.dumps(shown, indent=1), prediction=shown["prediction"],
                               evidence_json=json.dumps(shown["evidence"], indent=1))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat_text(tokenizer, msgs):
    """Apply the model's chat template; models without a system role get it prepended to the user turn."""
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}]
        return tokenizer.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


class TemplateBackend:
    name, revision = "template", "n/a"

    def generate(self, batch_msgs, batch_inputs):
        return [json.dumps(render_llm_input(inp)) for inp in batch_inputs]


class HFBackend:
    def __init__(self, model_id, dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(DECODING["seed"])
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype),
                                                          device_map="auto").eval()
        self.name = model_id
        self.revision = getattr(self.model.config, "_commit_hash", None) or "unknown"

    def generate(self, batch_msgs, batch_inputs):
        texts = [chat_text(self.tok, m) for m in batch_msgs]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=DECODING["max_new_tokens"],
                                      pad_token_id=self.tok.pad_token_id)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def vllm_sampling(schema, max_tokens=None):
    """Greedy, seeded sampling params with JSON-schema constrained decoding (API name differs by vLLM version)."""
    from vllm import SamplingParams
    kwargs = {"temperature": 0.0, "max_tokens": max_tokens or DECODING["max_new_tokens"], "seed": DECODING["seed"]}
    guided = False
    if schema is not None:
        try:
            from vllm.sampling_params import StructuredOutputsParams
            kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
            guided = True
        except ImportError:
            try:
                from vllm.sampling_params import GuidedDecodingParams
                kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
                guided = True
            except ImportError:
                pass
    return SamplingParams(**kwargs), guided


def vllm_engine(model_id, max_model_len=4096, **extra):
    """Mistral-format repos (tekken tokenizer only, e.g. Mistral Small 3.x) need vLLM's mistral tokenizer mode."""
    from vllm import LLM
    if "Mistral-Small" in model_id:
        extra.setdefault("tokenizer_mode", "mistral")
    extra.setdefault("tensor_parallel_size", int(os.environ.get("VLLM_TP", "1")))  # 32B on 40 GB A100s needs 2
    return LLM(model=model_id, seed=DECODING["seed"], dtype="bfloat16", max_model_len=max_model_len, **extra)


def vllm_chat(llm, model_id, conversations, params):
    """Chat API (each model's own template). Qwen3: thinking off. Falls back to merging the system prompt into
    the user turn for templates without a system role."""
    kwargs = {"chat_template_kwargs": {"enable_thinking": False}} if "Qwen3" in model_id else {}
    try:
        outs = llm.chat(conversations, params, use_tqdm=False, **kwargs)
    except Exception as err:
        if "system" not in str(err).lower():
            raise
        merged = [[{"role": "user", "content": c[0]["content"] + "\n\n" + c[1]["content"]}] for c in conversations]
        outs = llm.chat(merged, params, use_tqdm=False, **kwargs)
    return [o.outputs[0].text for o in outs]


class VLLMBackend:
    def __init__(self, model_id, guided=True, schema=None):
        self.llm = vllm_engine(model_id)
        self.params, self.guided = vllm_sampling((schema or li.OUTPUT_SCHEMA) if guided else None)
        self.name = model_id
        self.revision = "see HF cache snapshot"

    def generate(self, batch_msgs, batch_inputs):
        return vllm_chat(self.llm, self.name, batch_msgs, self.params)


def assemble(spec, inp, raw):
    """Full-output prompts: the model output as is. reasons_only prompts: code-built summary, caveats and
    limitations (template.render_llm_input) around the model's reason sentences."""
    if spec.get("mode") != "reasons_only":
        return raw
    final = render_llm_input(inp)
    parsed, ok = verifier.parse(raw)
    reasons = parsed.get("reasons") if ok and isinstance(parsed, dict) else None
    final["reasons"] = reasons if isinstance(reasons, list) else []
    labels = {e["feature_id"]: li.encoding_label(e) for e in inp["evidence"]}
    for r in final["reasons"]:  # encoding labels are code-added, so they hold whatever the LLM wrote
        if isinstance(r, dict) and labels.get(r.get("feature_id")):
            r["text"] = r.get("text", "").rstrip(" .") + labels[r["feature_id"]] + "."
    return final


def stratum(row, inp):
    correct = (inp["prediction"] == "malignant") == (row["label"] == 1)
    return f"{'correct' if correct else 'wrong'}|{'uncertain' if inp['uncertain'] else 'confident'}"


def main(opts):
    rows = [json.loads(line) for line in open(opts.inputs, encoding="utf-8")]
    threshold = li.entropy_threshold_from_val(rows)
    rows = [r for r in rows if r["split"] == opts.split]
    if opts.limit:
        rows = rows[:opts.limit]
    if opts.encoding:
        enc = [encodings.encode(r["llm_input"], r["mask_quality"], threshold, opts.encoding) for r in rows]
        prompt_inputs, inputs, allowed = (list(x) for x in zip(*enc)) if enc else ([], [], [])
    else:
        inputs = [li.build(r["llm_input"], r["mask_quality"], threshold) for r in rows]
        prompt_inputs, allowed = [None] * len(rows), [None] * len(rows)

    if opts.backend == "template":
        backend, spec = TemplateBackend(), {"version": "template_v1", "sha256_16": "n/a"}
    else:
        spec = load_prompt(opts.prompt)
        schema = li.REASONS_SCHEMA if spec.get("mode") == "reasons_only" else li.OUTPUT_SCHEMA
        backend = HFBackend(opts.model) if opts.backend == "hf" else VLLMBackend(opts.model, not opts.no_guided, schema)

    run_name = f"{backend.name.split('/')[-1]}__{spec['version']}" + (f"__{opts.encoding}" if opts.encoding else "")
    out = Path(opts.out) / run_name
    out.mkdir(parents=True, exist_ok=True)
    t0, results = time.time(), []
    with open(out / "generations.jsonl", "w", encoding="utf-8") as fh:
        for k in range(0, len(rows), opts.batch_size):
            sl = slice(k, k + opts.batch_size)
            batch_rows, batch_inp, batch_shown, batch_allowed = rows[sl], inputs[sl], prompt_inputs[sl], allowed[sl]
            batch_msgs = ([messages(spec, i, s) for i, s in zip(batch_inp, batch_shown)] if opts.backend != "template"
                          else [None] * len(batch_inp))
            for row, inp, ok_numbers, raw in zip(batch_rows, batch_inp, batch_allowed,
                                                 backend.generate(batch_msgs, batch_inp)):
                final = assemble(spec, inp, raw)
                check = verifier.verify(inp, final, ok_numbers)
                results.append((stratum(row, inp), check))
                fh.write(json.dumps({"isic_id": row["isic_id"], "row_id": row.get("row_id", row["isic_id"]),
                                     "variant": row.get("variant", "base"), "source": row["source"], "label": row["label"],
                                     "stratum": stratum(row, inp), "input_hash": li.input_hash(inp),
                                     "llm_input": inp, "raw_output": raw, "final_output": final,
                                     "verify": check}) + "\n")
            print(f"{min(k + opts.batch_size, len(rows))}/{len(rows)}", flush=True)
    elapsed = time.time() - t0

    strata = sorted({s for s, _ in results})
    summary = {"overall": verifier.summarize([c for _, c in results]),
               "by_stratum": {s: verifier.summarize([c for t, c in results if t == s]) for s in strata}}
    manifest = {"model": backend.name, "model_revision": backend.revision, "backend": opts.backend,
                "guided_json": getattr(backend, "guided", False), "prompt_version": spec["version"],
                "prompt_sha256_16": spec["sha256_16"], "decoding": DECODING,
                "llm_input_version": li.LLM_INPUT_VERSION, "phrase_bank": li.PHRASE_BANK_VERSION,
                "encoding": opts.encoding or "C-phrase (llm_input)",
                "entropy_threshold_val_p75": threshold, "inputs": str(opts.inputs),
                "inputs_sha256_16": hashlib.sha256(Path(opts.inputs).read_bytes()).hexdigest()[:16],
                "split": opts.split, "n": len(rows), "seconds": round(elapsed, 1),
                "seconds_per_explanation": round(elapsed / max(len(rows), 1), 3), "python": platform.python_version()}
    if opts.backend != "template":
        import torch
        manifest["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        # vLLM runs the model in a separate engine process, so this process's allocator sees nothing
        manifest["max_gpu_mem_gb"] = (None if opts.backend == "vllm" else
                                      round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0)
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "verify_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["overall"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--backend", choices=["template", "hf", "vllm"], default="vllm")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--prompt", default="cite_v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--no_guided", action="store_true", help="vLLM: disable JSON-schema constrained decoding")
    ap.add_argument("--out", default=str(HERE / "output"))
    ap.add_argument("--encoding", choices=encodings.ENCODINGS, default=None,
                    help="attribution encoding for the paper's RQ2 (use with --prompt enc_v1); default: llm_input")
    main(ap.parse_args())
