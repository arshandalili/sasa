import argparse
import json
import random
import statistics
import time
from types import SimpleNamespace

import torch
from openai import OpenAI
from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
from sae_bench.evals.autointerp.main import AutoInterp
from sae_bench.sae_bench_utils import activation_collection, dataset_utils
from transformer_lens import HookedTransformer

import paths
from configs import load_config
from eval.core import SAEBenchConfig
from eval.dictionaries import load_sae

SAES = ("gpt2_l7/topk_l0120_w24576", "gpt2_l7/batchtopk_l0120_w24576", "gpt2_l7/matryoshka_l0120_w24576",
        "gpt2_l7/sasa_r6_l0120_w24576")
SEEDS = (42, 43, 44)
JUDGE = "gpt-4o-mini"
SAEBENCH = AutoInterpEvalConfig()
ANSWER_TOKENS = {"explain": SAEBENCH.max_tokens_in_explanation, "score": SAEBENCH.max_tokens_in_prediction}
LLM_BATCH_SIZE = 32
MAX_CHUNK_TOKENS = 10_000_000  # estimated prompt tokens per Batch API job: OpenAI caps the tokens it queues per model
MAX_REQUESTS = 3
POLL_SECONDS = 60
EXPLANATION_PLACEHOLDER = "@@EXPLANATION@@"
OUT = paths.RESULTS / "autointerp"


class LatentUnits:
    def __init__(self, sae):
        self.sae, self.cfg, self.device, self.W_dec = sae, SAEBenchConfig(sae.cfg), sae.device, sae.W_dec

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)


class GroupUnits:
    # one unit per SASA group, ||sigma(h) D_k a_k(h)||: the norm of its decoded output in residual-stream units
    def __init__(self, sae):
        self.sae, self.cfg, self.device = sae, SAEBenchConfig(sae.cfg), sae.device
        blocks = sae.W_dec.detach().float().view(sae.n_groups, sae.group_rank, -1)
        self.gram = blocks @ blocks.transpose(-1, -2)
        self.W_dec = blocks[:, 0]  # SAEBench reads only its number of rows, the number of units

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        codes = self.sae.encode(x).float().view(-1, self.sae.n_groups, self.sae.group_rank)
        energy = torch.einsum("nki,kij,nkj->nk", codes, self.gram, codes).clamp(min=0)
        return (energy.sqrt() * self.sae.ln_std.reshape(-1, 1).float()).view(*x.shape[:-1], -1)


def prepare(names, seeds) -> None:
    torch.set_grad_enabled(False)
    config = load_config(names[0].split("/")[0])
    model = HookedTransformer.from_pretrained_no_processing(config["model_name"], device="cuda").eval()
    tokens_file = OUT / f"tokens_{SAEBENCH.total_tokens}_ctx{SAEBENCH.llm_context_size}.pt"
    if not tokens_file.exists():
        tokens_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset_utils.load_and_tokenize_dataset(SAEBENCH.dataset_name, SAEBENCH.llm_context_size,
                                                           SAEBENCH.total_tokens, model.tokenizer), tokens_file)
    tokens = torch.load(tokens_file).to("cuda")
    for name in names:
        sae_name = name.split("/")[1]
        todo = [seed for seed in seeds if not (OUT / sae_name / f"seed{seed}.json").exists()]
        if not todo:
            continue
        sae = load_sae(name)
        units = GroupUnits(sae) if hasattr(sae, "group_rank") else LatentUnits(sae)
        sparsity = activation_collection.get_feature_activation_sparsity(
            tokens, model, units, LLM_BATCH_SIZE, units.cfg.hook_layer, units.cfg.hook_name,
            mask_bos_pad_eos_tokens=True)
        for seed in todo:
            random.seed(seed)
            torch.manual_seed(seed)
            eval_config = AutoInterpEvalConfig(model_name=config["model_name"], llm_batch_size=LLM_BATCH_SIZE,
                                               llm_dtype="float32", random_seed=seed)
            judge = AutoInterp(cfg=eval_config, model=model, sae=units, tokenized_dataset=tokens,
                               sparsity=sparsity.clone(), device="cuda", api_key="")
            generation, scoring = judge.gather_data()
            records = [{"unit": unit, "generation": judge.get_generation_prompts(generation[unit]),
                        "scoring": judge.get_scoring_prompts(EXPLANATION_PLACEHOLDER, scoring[unit]),
                        "active": [example.is_active for example in scoring[unit]]} for unit in sorted(generation)]
            out = OUT / sae_name / f"seed{seed}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"sae": name, "seed": seed, "units": records}))
        del sae, units
        torch.cuda.empty_cache()


def stored_units(names, seeds):
    for name in names:
        sae_name = name.split("/")[1]
        for seed in seeds:
            for record in json.loads((OUT / sae_name / f"seed{seed}.json").read_text())["units"]:
                yield f"{sae_name}-{seed}-{record['unit']}", record


def read_answers(stage: str) -> dict:
    path, answers = OUT / f"{stage}.jsonl", {}
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a line cut short by a stopped run
            continue
        answers[row["id"]] = row
    return answers


def answered(answer: dict | None) -> bool:
    # an empty or filtered answer is asked again; an answer SAEBench cannot parse is final
    return answer is not None and answer["stop_reason"] != "content_filter" and answer["text"] != ""


def request(request_id: str, messages: list, max_tokens: int) -> dict:
    return {"custom_id": request_id, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": JUDGE, "messages": messages, "max_tokens": max_tokens}}


def chunks(requests: list) -> list[list]:
    out, size = [[]], 0.0
    for item in requests:
        tokens = len(json.dumps(item["body"]["messages"])) / 3
        if out[-1] and size + tokens > MAX_CHUNK_TOKENS:
            out.append([])
            size = 0.0
        out[-1].append(item)
        size += tokens
    return out


def submit(client, stage: str, requests: list) -> str:
    path = OUT / "requests" / f"{stage}_{time.strftime('%m%d_%H%M%S')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item) + "\n" for item in requests))
    with open(path, "rb") as handle:
        upload = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions", completion_window="24h")
    with open(OUT / f"{stage}_batches.txt", "a") as handle:  # a stopped run collects it instead of submitting again
        handle.write(batch.id + "\n")
    return batch.id


def collect(client, batch_id: str, stage: str) -> None:
    collected = OUT / f"{stage}_collected.txt"
    if collected.exists() and batch_id in collected.read_text().split():
        return
    batch = client.batches.retrieve(batch_id)
    while batch.status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(POLL_SECONDS)
        batch = client.batches.retrieve(batch_id)
    if batch.status == "failed":
        raise SystemExit(f"batch {batch_id} failed: {batch.errors}")
    results = client.files.content(batch.output_file_id).text.splitlines() if batch.output_file_id else []
    path = OUT / f"{stage}.jsonl"
    with open(path, "a") as answers:
        if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
            answers.write("\n")
        for result in map(json.loads, results):
            response = result.get("response") or {}
            if response.get("status_code") == 200:
                choice = response["body"]["choices"][0]
                text = (choice["message"]["content"] or "").strip()
                answers.write(json.dumps({"id": result["custom_id"], "text": text,
                                          "stop_reason": choice["finish_reason"]}) + "\n")
    with open(collected, "a") as handle:
        handle.write(batch_id + "\n")


def judge(stage: str, names, seeds) -> None:
    client = OpenAI()
    submitted = OUT / f"{stage}_batches.txt"
    for batch_id in submitted.read_text().split() if submitted.exists() else []:
        collect(client, batch_id, stage)
    for _ in range(MAX_REQUESTS):
        answers, explanations = read_answers(stage), read_answers("explain")
        requests = []
        for request_id, record in stored_units(names, seeds):
            if answered(answers.get(request_id)):
                continue
            if stage == "explain":
                requests.append(request(request_id, record["generation"], ANSWER_TOKENS["explain"]))
            elif answered(explanations.get(request_id)):
                explanation = AutoInterp.parse_explanation(None, explanations[request_id]["text"])
                messages = [dict(message, content=message["content"].replace(EXPLANATION_PLACEHOLDER, explanation))
                            for message in record["scoring"]]
                requests.append(request(request_id, messages, ANSWER_TOKENS["score"]))
        if not requests:
            break
        for chunk in chunks(requests):
            collect(client, submit(client, stage, chunk), stage)


def write_results(names, seeds) -> None:
    explanations, predictions = read_answers("explain"), read_answers("score")
    scores = {}
    for request_id, record in stored_units(names, seeds):
        name, seed, _ = request_id.rsplit("-", 2)
        prediction = predictions.get(request_id)
        if not (answered(explanations.get(request_id)) and answered(prediction)):
            continue
        predicted = AutoInterp.parse_predictions(None, prediction["text"])
        if predicted is not None:  # SAEBench drops a unit whose answer it cannot parse
            examples = [SimpleNamespace(is_active=active) for active in record["active"]]
            scores.setdefault(name, {}).setdefault(seed, []).append(
                AutoInterp.score_predictions(None, predicted, examples))
    results = {}
    for name, per_seed in scores.items():
        means = [100 * statistics.mean(values) for values in per_seed.values()]
        results[name] = {"mean": statistics.mean(means), "sd": statistics.stdev(means) if len(means) > 1 else 0.0}
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "explain", "score", "results"))
    parser.add_argument("--saes", nargs="+", default=list(SAES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.saes, args.seeds)
    elif args.stage == "results":
        write_results(args.saes, args.seeds)
    else:
        judge(args.stage, args.saes, args.seeds)


if __name__ == "__main__":
    main()
