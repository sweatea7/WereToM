from __future__ import annotations

import argparse

import csv

import json

import random

import time

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from pathlib import Path

from threading import Lock

from typing import Any, Dict, Iterable, List, Optional, Tuple


from openai import OpenAI


import rerank_vote_predict_dataset as rerank


OUTPUT_DIR = Path(__file__).resolve().parent


SOURCE_CHECKPOINT = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_rerank.checkpoint.jsonl"

REPAIR_CHECKPOINT = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_drop_lt60_repair_60_80.checkpoint.jsonl"

REPAIR_ERROR_LOG = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_drop_lt60_repair_60_80.errors.jsonl"

REPAIR_OUTPUT = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_drop_lt60_repair_60_80.json"

REPAIR_SUMMARY = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_drop_lt60_repair_60_80.summary.json"

REMAINING_LOW_CSV = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_drop_lt60_repair_60_80.remaining_low.csv"

DROPPED_LOW_CSV = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_drop_lt60.discarded.csv"


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Repair low-score reranked werewolf dataset records.")

    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT, help="Full rerank checkpoint JSONL.")

    parser.add_argument("--repair-checkpoint", type=Path, default=REPAIR_CHECKPOINT, help="Append-only repair checkpoint JSONL.")

    parser.add_argument("--error-log", type=Path, default=REPAIR_ERROR_LOG, help="Repair error log JSONL.")

    parser.add_argument("--output", type=Path, default=REPAIR_OUTPUT, help="Merged repaired dataset JSON.")

    parser.add_argument("--summary", type=Path, default=REPAIR_SUMMARY, help="Repair summary JSON.")

    parser.add_argument("--remaining-low-csv", type=Path, default=REMAINING_LOW_CSV, help="CSV of records still below threshold.")

    parser.add_argument("--dropped-low-csv", type=Path, default=DROPPED_LOW_CSV, help="CSV audit list of dropped records.")


    parser.add_argument("--score-min", type=float, default=60.0, help="Only repair records with selected score >= this value.")

    parser.add_argument("--score-max", type=float, default=80.0, help="Only repair records with selected score < this value.")

    parser.add_argument("--accept-threshold", type=float, default=None, help="Override acceptance threshold for all repaired records.")

    parser.add_argument("--accept-threshold-60-70", type=float, default=80.0, help="Default acceptance threshold for [60, 70) records.")

    parser.add_argument("--accept-threshold-70-80", type=float, default=85.0, help="Default acceptance threshold for [70, 80) records.")

    parser.add_argument("--drop-below", type=float, default=60.0, help="Drop records below this score from the merged output.")

    parser.add_argument("--min-improvement", type=float, default=0.01, help="Repair must improve over current score by this margin.")

    parser.add_argument("--limit", type=int, default=None, help="Small experiment size after score filtering.")

    parser.add_argument("--target-offset", type=int, default=0, help="Skip first N filtered targets after sorting.")

    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N filtered targets.")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--order", choices=["score", "index"], default="score", help="Target processing order.")


    parser.add_argument("--candidates", type=int, default=2, help="Repair candidates per low-score record.")

    parser.add_argument("--workers", type=int, default=2, help="Sample-level concurrency.")

    parser.add_argument("--temperature", type=float, default=0.4, help="Repair generation temperature.")

    parser.add_argument("--max-tokens", type=int, default=750)


    parser.add_argument("--generator-model", default=rerank.GENERATOR_MODEL)

    parser.add_argument("--generator-base-url", default=rerank.GENERATOR_BASE_URL)

    parser.add_argument("--generator-api-key", default=rerank.GENERATOR_API_KEY)

    parser.add_argument("--judge-model", default=rerank.JUDGE_MODEL)

    parser.add_argument("--judge-base-url", default=rerank.JUDGE_BASE_URL)

    parser.add_argument("--judge-api-key", default=rerank.JUDGE_API_KEY)

    parser.add_argument("--no-llm-judge", action="store_true", help="Use local score only. Not recommended for final repair.")

    parser.add_argument("--allow-local-fallback", action="store_true", help="Do not stop when judge fails; use local score.")

    parser.add_argument("--dry-run", action="store_true", help="Only list repair targets; do not call APIs or write merged output.")

    parser.add_argument("--keep-meta", action="store_true", help="Keep _repair_meta in accepted final records.")

    return parser.parse_args()


def append_jsonl(path: Path, record: Dict[str, Any], lock: Optional[Lock] = None) -> None:

    def write() -> None:

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as file:

            file.write(json.dumps(record, ensure_ascii=False) + "\n")


    if lock:

        with lock:

            write()

    else:

        write()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    if not path.exists():

        return rows

    with path.open("r", encoding="utf-8", errors="replace") as file:

        for line in file:

            if not line.strip():

                continue

            try:

                rows.append(json.loads(line))

            except Exception:

                continue

    return rows


def write_json(path: Path, data: Any) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:

        json.dump(data, file, ensure_ascii=False, indent=2)


def selected_score_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:

    source = record.get("selected_source")

    for score in record.get("scores") or []:

        if score.get("source") == source:

            return score

    return None


def score_value(score: Optional[Dict[str, Any]], default: float = -1.0) -> float:

    if not score:

        return default

    try:

        return float(score.get("final_score", default))

    except (TypeError, ValueError):

        return default


def load_source_records(path: Path) -> Dict[int, Dict[str, Any]]:

    records: Dict[int, Dict[str, Any]] = {}

    for row in read_jsonl(path):

        if "index" in row:

            records[int(row["index"])] = row

    return records


def load_repair_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:

    repairs: Dict[int, Dict[str, Any]] = {}

    for row in read_jsonl(path):

        if "index" in row:

            repairs[int(row["index"])] = row

    return repairs


def target_records(source_records: Dict[int, Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:

    targets = []

    for index, record in source_records.items():

        if record.get("status") == "skipped" or record.get("selected_source") is None:

            continue

        score = selected_score_record(record)

        selected_score = score_value(score)

        if args.score_min <= selected_score < args.score_max:

            targets.append(

                {

                    "index": index,

                    "record": record,

                    "selected_score": selected_score,

                }

            )

    if args.order == "score":

        targets.sort(key=lambda row: (row["selected_score"], row["index"]))

    else:

        targets.sort(key=lambda row: row["index"])


    if args.target_offset:

        targets = targets[max(0, args.target_offset) :]

    if args.sample is not None and args.sample < len(targets):

        rng = random.Random(args.seed)

        targets = sorted(rng.sample(targets, args.sample), key=lambda row: row["index"])

    if args.limit is not None:

        targets = targets[: max(0, args.limit)]

    return targets


def compact_repair_diagnosis(score: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    """Expose repair hints without leaking hidden-role ground truth to the generator."""

    if not score:

        return {}

    local = score.get("local") or {}

    llm = score.get("llm") or {}

    return {

        "final_score": score.get("final_score"),

        "local_score": local.get("score"),

        "local_accuracy": local.get("accuracy"),

        "local_consistency": local.get("consistency"),

        "local_consistency_issues": (local.get("consistency_details") or {}).get("issues"),

        "judge_result_accuracy": llm.get("result_accuracy"),

        "judge_consistency": llm.get("consistency"),

        "judge_total": llm.get("total"),

        "repair_instruction": (

            "请根据游戏输入独立修正投票与身份预测；"

            "评分数值仅表示当前答案存在提升空间，不提供真实隐藏身份答案。"

        ),

    }


def acceptance_threshold(base_score: float, args: argparse.Namespace) -> float:

    if args.accept_threshold is not None:

        return float(args.accept_threshold)

    if base_score < 70.0:

        return float(args.accept_threshold_60_70)

    return float(args.accept_threshold_70_80)


def build_repair_messages(

    item: Dict[str, Any],

    current_score: Optional[Dict[str, Any]],

    candidate_no: int,

    accept_threshold: float,

) -> List[Dict[str, str]]:

    system = (

        "你是狼人杀心智理论数据集的低分样本修复模型。"

        "你的任务不是重新解释评分，而是直接产出一条更高质量的训练样本 output。"

        "必须保持第一视角，严格使用 <think>...</think> 和 <answer>...</answer>。"

        "think 内必须是合法 JSON，包含 ToM1、ToM2、Strategy 三个字段。"

        "answer 内必须先明确投票目标，再给出其他存活玩家身份预测 JSON。"

        "身份只能使用：平民、狼人、预言家、女巫、猎人。"

        "不要预测自己，不要预测已死亡玩家，不要输出解释性前后缀。"

        "总字数尽量控制在 400 字以内。"

        "你只能依据玩家在 input 中可获得的信息进行推断，不得假设知道其他玩家真实底牌。"

    )

    payload = {

        "repair_goal": f"修复当前低分 output，目标综合分至少达到 {accept_threshold}。",

        "candidate_no": candidate_no,

        "key_requirements": [

            "投票目标必须符合玩家真实阵营收益。",

            "身份预测 JSON 应根据 input 尽量准确判断身份，且只包含其他存活玩家。",

            "ToM1 必须准确概括 input 中的场上局势，并符合自己的真实底牌视角。",

            "Strategy 中的战术必须在 answer 的投票目标中被正确执行。",

            "如当前判官理由指出投票错误、身份预测错误、局势概括错误或格式问题，请优先修复这些问题。",

        ],

        "input": item.get("input", ""),

        "current_output": item.get("output", ""),

        "current_score_and_non_leaking_diagnosis": compact_repair_diagnosis(current_score),

    }

    user = (

        "请基于以下 JSON 信息，直接输出修复后的 output。"

        "只能输出 <think>...</think><answer>...</answer>，不要输出任何解释。\n"

        + json.dumps(payload, ensure_ascii=False)

    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_repair_candidates(

    client: OpenAI,

    item: Dict[str, Any],

    current_score: Optional[Dict[str, Any]],

    args: argparse.Namespace,

) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:

    candidates: List[Dict[str, Any]] = []

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for candidate_no in range(1, args.candidates + 1):

        content, usage = rerank.retry_call(

            rerank.call_chat,

            client,

            args.generator_model,

            build_repair_messages(item, current_score, candidate_no, args.accept_threshold),

            args.temperature,

            args.max_tokens,

        )

        for key, value in usage.items():

            usage_total[key] += value

        candidates.append({"source": f"repair_{candidate_no}", "output": content})

    return candidates, usage_total


def judge_repair_candidate(

    judge_client: Optional[OpenAI],

    args: argparse.Namespace,

    item: Dict[str, Any],

    candidate: Dict[str, Any],

    use_llm_judge: bool,

) -> Dict[str, Any]:

    return rerank.judge_candidate(judge_client, args.judge_model, item, candidate, use_llm_judge)


def process_repair_one(index: int, source_record: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:

    current_item = dict(source_record["selected_item"])

    current_score = selected_score_record(source_record)

    current_final = score_value(current_score)

    required_score = acceptance_threshold(current_final, args)

    generator_client = OpenAI(api_key=args.generator_api_key, base_url=args.generator_base_url)

    use_llm_judge = bool(args.judge_api_key and args.judge_model and not args.no_llm_judge)

    judge_client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url) if use_llm_judge else None


    repair_args = argparse.Namespace(**{**vars(args), "accept_threshold": required_score})

    repair_candidates, generation_usage = generate_repair_candidates(generator_client, current_item, current_score, repair_args)


    scored_repairs = []

    judge_usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for candidate in repair_candidates:

        try:

            judgement = judge_repair_candidate(judge_client, args, current_item, candidate, use_llm_judge)

        except Exception as exc:

            if rerank.is_sensitive_judge_error(exc):

                judgement = {

                    "local": rerank.local_score(current_item, candidate["output"]),

                    "llm": None,

                    "final_score": -1.0,

                    "judge_usage": {},

                    "judge_error": repr(exc),

                }

            elif args.allow_local_fallback:

                local = rerank.local_score(current_item, candidate["output"])

                judgement = {

                    "local": local,

                    "llm": None,

                    "final_score": local["score"],

                    "judge_usage": {},

                    "judge_error": repr(exc),

                }

            else:

                raise

        for key, value in judgement.get("judge_usage", {}).items():

            judge_usage_total[key] += value

        scored_repairs.append({**candidate, "judgement": judgement})


    best_repair = max(scored_repairs, key=lambda row: row["judgement"]["final_score"], default=None)

    best_repair_score = best_repair["judgement"]["final_score"] if best_repair else -1.0

    accepted = bool(

        best_repair

        and best_repair_score >= required_score

        and best_repair_score > current_final + args.min_improvement

    )


    selected_item = dict(current_item)

    if accepted:

        selected_item["output"] = best_repair["output"]

        selected_source = best_repair["source"]

        selected_score = best_repair_score

    else:

        selected_source = "base_selected"

        selected_score = current_final


    if args.keep_meta:

        selected_item["_repair_meta"] = {

            "accepted": accepted,

            "selected_source": selected_source,

            "base_score": current_final,

            "selected_score": selected_score,

            "required_score": required_score,

            "source_checkpoint_index": index,

        }


    return {

        "index": index,

        "item_hash": source_record.get("item_hash") or rerank.stable_hash(current_item),

        "status": "accepted" if accepted else "kept",

        "accepted": accepted,

        "base_selected_source": source_record.get("selected_source"),

        "base_score": current_final,

        "required_score": required_score,

        "selected_source": selected_source,

        "selected_score": selected_score,

        "selected_item": selected_item,

        "scores": [

            {

                "source": "base_selected",

                "final_score": current_final,

                "local": (current_score or {}).get("local"),

                "llm": (current_score or {}).get("llm"),

                "judge_error": (current_score or {}).get("judge_error"),

            }

        ]

        + [

            {

                "source": candidate["source"],

                "final_score": candidate["judgement"]["final_score"],

                "local": candidate["judgement"]["local"],

                "llm": candidate["judgement"].get("llm"),

                "judge_error": candidate["judgement"].get("judge_error"),

            }

            for candidate in scored_repairs

        ],

        "usage": {"generation": generation_usage, "judge": judge_usage_total},

    }


def merge_output(

    source_records: Dict[int, Dict[str, Any]],

    repair_records: Dict[int, Dict[str, Any]],

    args: argparse.Namespace,

) -> List[Dict[str, Any]]:

    output_data: List[Dict[str, Any]] = []

    for index in sorted(source_records):

        source = source_records[index]

        if source.get("status") == "skipped" or source.get("selected_source") is None:

            continue

        base_score = score_value(selected_score_record(source))

        if base_score < args.drop_below:

            continue

        repair = repair_records.get(index)

        if repair and repair.get("accepted") and repair.get("selected_item"):

            output_data.append(repair["selected_item"])

        else:

            output_data.append(source["selected_item"])

    write_json(args.output, output_data)

    return output_data


def summarize_records(

    source_records: Dict[int, Dict[str, Any]],

    repair_records: Dict[int, Dict[str, Any]],

    output_data: List[Dict[str, Any]],

    args: argparse.Namespace,

) -> Dict[str, Any]:

    accepted = [row for row in repair_records.values() if row.get("accepted")]

    attempted = [row for row in repair_records.values() if row.get("status") in {"accepted", "kept"}]

    gains = [float(row.get("selected_score", 0)) - float(row.get("base_score", 0)) for row in accepted]

    remaining_low = []

    dropped_records = []

    for index, source in source_records.items():

        if source.get("status") == "skipped" or source.get("selected_source") is None:

            continue

        base_score = score_value(selected_score_record(source))

        if base_score < args.drop_below:

            dropped_records.append(

                {

                    "index": index,

                    "base_score": round(base_score, 3),

                    "selected_source": source.get("selected_source"),

                    "item_hash": source.get("item_hash", ""),

                }

            )

            continue

        repair = repair_records.get(index)

        final_score = float(repair.get("selected_score")) if repair else base_score

        if final_score < args.score_max:

            remaining_low.append(

                {

                    "index": index,

                    "final_score": round(final_score, 3),

                    "base_score": round(base_score, 3),

                    "repair_status": repair.get("status") if repair else "not_repaired",

                    "selected_source": repair.get("selected_source") if repair else source.get("selected_source"),

                    "item_hash": source.get("item_hash", ""),

                }

            )


    with args.remaining_low_csv.open("w", newline="", encoding="utf-8-sig") as file:

        writer = csv.DictWriter(

            file,

            fieldnames=["index", "final_score", "base_score", "repair_status", "selected_source", "item_hash"],

        )

        writer.writeheader()

        writer.writerows(sorted(remaining_low, key=lambda row: row["final_score"]))


    with args.dropped_low_csv.open("w", newline="", encoding="utf-8-sig") as file:

        writer = csv.DictWriter(file, fieldnames=["index", "base_score", "selected_source", "item_hash"])

        writer.writeheader()

        writer.writerows(sorted(dropped_records, key=lambda row: row["base_score"]))


    usage = rerank.summarize_usage(attempted)

    summary = {

        "source_checkpoint": str(args.source_checkpoint),

        "repair_checkpoint": str(args.repair_checkpoint),

        "output": str(args.output),

        "score_min": args.score_min,

        "score_max": args.score_max,

        "accept_threshold_override": args.accept_threshold,

        "accept_threshold_60_70": args.accept_threshold_60_70,

        "accept_threshold_70_80": args.accept_threshold_70_80,

        "drop_below": args.drop_below,

        "source_records": len(source_records),

        "output_records": len(output_data),

        "dropped_below_threshold_records": len(dropped_records),

        "dropped_low_csv": str(args.dropped_low_csv),

        "repair_attempted_records": len(attempted),

        "repair_accepted_records": len(accepted),

        "repair_kept_records": len(attempted) - len(accepted),

        "average_gain_on_accepted": round(sum(gains) / len(gains), 3) if gains else 0.0,

        "remaining_below_score_max": len(remaining_low),

        "remaining_low_csv": str(args.remaining_low_csv),

        "usage": usage,

    }

    write_json(args.summary, summary)

    return summary


def main() -> None:

    args = parse_args()

    source_records = load_source_records(args.source_checkpoint)

    repair_records = load_repair_checkpoint(args.repair_checkpoint)

    eligible_targets = target_records(

        source_records,

        argparse.Namespace(**{**vars(args), "target_offset": 0, "limit": None, "sample": None}),

    )

    targets = target_records(source_records, args)

    targets = [row for row in targets if row["index"] not in repair_records]


    print(f"source checkpoint records: {len(source_records)}")

    print(f"repair checkpoint loaded: {len(repair_records)}")

    print(f"eligible records in score window: {len(eligible_targets)}")

    print(f"target records this run: {len(targets)}")

    print(f"score window: [{args.score_min}, {args.score_max})")

    print(f"drop below: {args.drop_below}")

    if args.accept_threshold is not None:

        print(f"accept threshold override: {args.accept_threshold}")

    else:

        print(f"accept thresholds: [60,70) >= {args.accept_threshold_60_70}; [70,80) >= {args.accept_threshold_70_80}")

    print(f"repair candidates per item: {args.candidates}")

    print(f"workers: {args.workers}")


    if args.dry_run:

        preview = [

            {"index": row["index"], "selected_score": round(row["selected_score"], 3)}

            for row in targets[:20]

        ]

        print(json.dumps({"preview_first20": preview}, ensure_ascii=False, indent=2))

        return


    if not args.generator_api_key:

        raise RuntimeError("Missing generator API key.")

    if not args.no_llm_judge and (not args.judge_api_key or not args.judge_model):

        raise RuntimeError("Missing judge API key/model. Use --no-llm-judge only for debugging.")


    lock = Lock()

    failed = False

    workers = max(1, int(args.workers or 1))

    pending = {}

    next_pos = 0

    completed_this_run = 0


    with ThreadPoolExecutor(max_workers=workers) as executor:

        while next_pos < len(targets) and len(pending) < workers:

            row = targets[next_pos]

            print(f"[{next_pos + 1}/{len(targets)}] start index={row['index']} score={row['selected_score']:.3f}", flush=True)

            future = executor.submit(process_repair_one, row["index"], row["record"], args)

            pending[future] = row

            next_pos += 1


        while pending:

            done, _ = wait(pending, return_when=FIRST_COMPLETED)

            for future in done:

                row = pending.pop(future)

                index = row["index"]

                try:

                    record = future.result()

                except Exception as exc:

                    append_jsonl(

                        args.error_log,

                        {

                            "index": index,

                            "item_hash": row["record"].get("item_hash"),

                            "stage": "repair_one",

                            "error": repr(exc),

                        },

                        lock,

                    )

                    print(f"[{completed_this_run + 1}/{len(targets)}] failed index={index}: {exc!r}", flush=True)

                    failed = True

                    break


                append_jsonl(args.repair_checkpoint, record, lock)

                repair_records[index] = record

                completed_this_run += 1

                print(

                    f"[{completed_this_run}/{len(targets)}] done index={index}: "

                    f"status={record['status']} base={record['base_score']:.3f} selected={record['selected_score']:.3f}",

                    flush=True,

                )


                if next_pos < len(targets):

                    row2 = targets[next_pos]

                    print(

                        f"[{next_pos + 1}/{len(targets)}] start index={row2['index']} score={row2['selected_score']:.3f}",

                        flush=True,

                    )

                    future2 = executor.submit(process_repair_one, row2["index"], row2["record"], args)

                    pending[future2] = row2

                    next_pos += 1


            if failed:

                break


    if failed:

        print("repair stopped because of an API or processing error. Completed records remain in repair checkpoint.")

        return


    output_data = merge_output(source_records, repair_records, args)

    summary = summarize_records(source_records, repair_records, output_data, args)

    print(f"saved repaired dataset: {args.output}")

    print(f"saved repair summary: {args.summary}")

    print(f"saved remaining low CSV: {args.remaining_low_csv}")

    print(f"saved dropped low CSV: {args.dropped_low_csv}")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":

    main()
