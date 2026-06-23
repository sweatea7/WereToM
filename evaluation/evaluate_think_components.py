import argparse

import hashlib

import json

import os

import re

import threading

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from typing import Any, Dict, List, Optional


from openai import OpenAI


try:

    from tqdm import tqdm

except ImportError:

    tqdm = None


OUTPUT_BASE_DIR = os.getenv("WERETOM_RESULT_DIR", "experiment_results")

RESULT_FILE_NAME = "results_vote_role_predict.json"

EVAL_NAME = "think_components"

EVAL_RESULT_FILE_NAME = f"{EVAL_NAME}.results.json"

EVAL_PARTIAL_FILE_NAME = f"{EVAL_NAME}.partial.json"

EVAL_CHECKPOINT_FILE_NAME = f"{EVAL_NAME}.checkpoint.jsonl"

SUMMARY_FILE_PATH = Path(OUTPUT_BASE_DIR) / f"{EVAL_NAME}.summary.json"

SELECTED_MODEL_DIRS = [

    "DeepSeek-V4-Flash",

    "DeepSeek-chat",

    "Qwen-Max",

    "Qwen3.5-flash",

    "doubao-pro",

    "doubao-seed",

    "ERNIE-4.5-Turbo",

    "GPT-5.4-Medium",

    "Gemini-2.5-Flash-Thinking",

    "GPT-5.5-High",

    "Qwen3-4B",

    "Qwen3-4B-SFT",

    "Qwen3-8B",

    "Qwen3-8B-SFT",

    "Gemma-2-9B",

    "Gemma-2-9B-SFT",

]


JUDGE_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

JUDGE_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

JUDGE_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

REQUEST_TIMEOUT = 45

MAX_RETRIES = 2

MAX_WORKERS = 32

VERBOSE_OUTPUT = False

CACHE_VERSION = "think_components_llm_eval"

CACHE_PATH = Path(os.getenv("WERETOM_CACHE_DIR", OUTPUT_BASE_DIR)) / "think_components_evaluation.cache.json"

ERROR_LOG_PATH = Path(os.getenv("WERETOM_CACHE_DIR", OUTPUT_BASE_DIR)) / "think_components_evaluation.errors.jsonl"


JUDGE_JSON_SYSTEM_PROMPT = (

    "You must output a valid json object only. "

    "Return strict JSON with double-quoted keys and string values where appropriate. "

    "Do not output markdown, code fences, explanations, or any text outside the json object."

)


def parse_args():

    parser = argparse.ArgumentParser(

        description="Evaluate ToM1, ToM2, and Strategy sections with one LLM-judge call per sample."

    )

    parser.add_argument("--verbose", action="store_true", help="Print detailed sub-metrics.")

    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Thread count.")

    return parser.parse_args()


def resolve_verbose(args):

    return VERBOSE_OUTPUT or bool(getattr(args, "verbose", False))


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:

    return max(low, min(high, float(value)))


def format_seconds(seconds):

    seconds = max(0, int(seconds))

    hours, rem = divmod(seconds, 3600)

    minutes, secs = divmod(rem, 60)

    if hours > 0:

        return f"{hours}h {minutes}m {secs}s"

    if minutes > 0:

        return f"{minutes}m {secs}s"

    return f"{secs}s"


def extract_tag(text, tag):

    match = re.search(fr"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL | re.IGNORECASE)

    return match.group(1).strip() if match else ""


def strip_code_fence(text):

    text = (text or "").strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_think_json(output_text):

    think = strip_code_fence(extract_tag(output_text, "think"))

    if not think:

        return {}

    match = re.search(r"\{.*\}", think, re.DOTALL)

    if match:

        think = match.group(0)

    try:

        parsed = json.loads(think)

    except Exception:

        return {}

    if not isinstance(parsed, dict):

        return {}

    return {str(k): str(v) for k, v in parsed.items()}


def extract_answer(output_text):

    return extract_tag(output_text, "answer") or (output_text or "").strip()


def load_cache():

    if not CACHE_PATH.exists():

        return {}

    try:

        with CACHE_PATH.open("r", encoding="utf-8") as f:

            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except Exception:

        return {}


def save_cache(cache):

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:

        json.dump(cache, f, ensure_ascii=False, separators=(",", ":"))

    os.replace(tmp_path, CACHE_PATH)


def atomic_write_json(path, data):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as f:

        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, path)


def append_jsonl(path, record):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:

        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_checkpoint(path):

    path = Path(path)

    completed = {}

    if not path.exists():

        return completed

    with path.open("r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            try:

                record = json.loads(line)

            except Exception:

                continue

            if "index" in record and "result" in record:

                completed[int(record["index"])] = record["result"]

    return completed


def build_ordered_results(completed, total_count):

    return [completed[idx] for idx in range(total_count) if idx in completed]


def load_result_list(path):

    path = Path(path)

    if not path.exists():

        return None

    try:

        with path.open("r", encoding="utf-8") as f:

            data = json.load(f)

        return data if isinstance(data, list) else None

    except Exception:

        return None


def append_error_log(record):

    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with ERROR_LOG_PATH.open("a", encoding="utf-8") as f:

        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_cache_key(model_name, item, output_text):

    payload = {

        "version": CACHE_VERSION,

        "task": "think_components_evaluation",

        "model_name": model_name,

        "input": item.get("input", ""),

        "player_id": item.get("player_id"),

        "player_role": item.get("player_role"),

        "output_text": output_text,

        "judge_model": JUDGE_MODEL,

    }

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_zero_record(reason: str, think: Optional[Dict[str, str]] = None):

    think = think or {}

    return {

        "__status__": "invalid_local",

        "valid": False,

        "tom1": {

            "grounding": 0.0,

            "coverage": 0.0,

            "perspective_alignment": 0.0,

            "conciseness": 0.0,

            "score": 0.0,

            "summary": reason,

        },

        "tom2": {

            "mental_modeling": 0.0,

            "consistency": 0.0,

            "specificity": 0.0,

            "risk_awareness": 0.0,

            "score": 0.0,

            "summary": reason,

        },

        "strategy": {

            "actionability": 0.0,

            "coherence": 0.0,

            "prioritization": 0.0,

            "risk_control": 0.0,

            "score": 0.0,

            "summary": reason,

        },

        "final_score": 0.0,

        "overall_summary": reason,

        "meta": {

            "think_keys": sorted(think.keys()),

        },

        "saved_at": int(time.time()),

        "error": reason,

    }


def build_judge_prompt(item: Dict[str, Any], think: Dict[str, str], answer_text: str) -> str:

    payload = {

        "player_id": item.get("player_id"),

        "player_role": item.get("player_role"),

        "input": item.get("input", ""),

        "tom1": think.get("ToM1", ""),

        "tom2": think.get("ToM2", ""),

        "strategy": think.get("Strategy", ""),

        "answer": answer_text,

    }

    return (

        "你是狼人杀心智理论数据的严格评测判官。现在你要同时评测 think 中的 ToM1、ToM2、Strategy 三个部分。\n"

        "评测原则：\n"

        "1. 只基于 input、think 和 answer 本身评分，不要代入额外事实。\n"

        "2. ToM1 评估的是对当前局势和公开信息的概括质量。\n"

        "3. ToM2 评估的是对其他玩家心理、动机、博弈关系的推断质量。\n"

        "4. Strategy 评估的是策略的可执行性、与前文一致性、优先级与风险控制。\n"

        "5. 如果内容编造、明显脱离视角、空泛套话严重、与 answer 或上下文矛盾，应降低分数。\n"

        "6. 你必须输出严格 JSON，不要输出任何额外文字。\n\n"

        "请返回如下结构：\n"

        "{\n"

        '  "tom1": {"grounding": 0-100, "coverage": 0-100, "perspective_alignment": 0-100, "conciseness": 0-100, "score": 0-100, "summary": "一句中文简评"},\n'

        '  "tom2": {"mental_modeling": 0-100, "consistency": 0-100, "specificity": 0-100, "risk_awareness": 0-100, "score": 0-100, "summary": "一句中文简评"},\n'

        '  "strategy": {"actionability": 0-100, "coherence": 0-100, "prioritization": 0-100, "risk_control": 0-100, "score": 0-100, "summary": "一句中文简评"},\n'

        '  "final_score": 0-100,\n'

        '  "overall_summary": "一句中文总评"\n'

        "}\n\n"

        "维度解释：\n"

        "- ToM1.grounding：是否建立在 input 已给事实之上。\n"

        "- ToM1.coverage：是否覆盖关键公开信息、票型、冲突与局势。\n"

        "- ToM1.perspective_alignment：是否符合当前玩家视角。\n"

        "- ToM1.conciseness：是否紧凑清晰、避免空话。\n"

        "- ToM2.mental_modeling：是否真正建模他人想法、动机、博弈。\n"

        "- ToM2.consistency：是否与 input / ToM1 保持一致。\n"

        "- ToM2.specificity：是否具体到玩家、关系和冲突，而非模板话。\n"

        "- ToM2.risk_awareness：是否体现出对误判、倒钩、冲锋、抗推等风险的理解。\n"

        "- Strategy.actionability：是否能直接指导后续回答或行动。\n"

        "- Strategy.coherence：是否与 ToM1 / ToM2 / input 一致。\n"

        "- Strategy.prioritization：是否明确主目标和次目标，优先级清楚。\n"

        "- Strategy.risk_control：是否考虑收益、暴露、站边和轮次风险。\n"

        "- final_score：综合三部分质量，不要简单取平均；若某一块明显短板，可拉低总分。\n\n"

        + json.dumps(payload, ensure_ascii=False)

    )


def call_judge_api(client, prompt):

    response = client.chat.completions.create(

        model=JUDGE_MODEL,

        messages=[

            {"role": "system", "content": JUDGE_JSON_SYSTEM_PROMPT},

            {"role": "user", "content": prompt},

        ],

        temperature=0.0,

        max_tokens=1100,

        timeout=REQUEST_TIMEOUT,

        response_format={"type": "json_object"},

    )

    return response.choices[0].message.content or ""


def parse_component(component: Dict[str, Any], required_keys: List[str], summary_key: str = "summary") -> Dict[str, Any]:

    parsed = {}

    for key in required_keys:

        parsed[key] = clamp(component.get(key, 0))

    parsed[summary_key] = str(component.get(summary_key, ""))

    return parsed


def parse_judge_response(raw_text):

    if not raw_text or not str(raw_text).strip():

        raise ValueError("Empty judge response")

    text = strip_code_fence(raw_text)

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:

        text = match.group(0)

    parsed = json.loads(text)

    if not isinstance(parsed, dict):

        raise ValueError("Judge response is not a JSON object")


    tom1 = parse_component(

        parsed.get("tom1", {}),

        ["grounding", "coverage", "perspective_alignment", "conciseness", "score"],

    )

    tom2 = parse_component(

        parsed.get("tom2", {}),

        ["mental_modeling", "consistency", "specificity", "risk_awareness", "score"],

    )

    strategy = parse_component(

        parsed.get("strategy", {}),

        ["actionability", "coherence", "prioritization", "risk_control", "score"],

    )

    final_score = clamp(parsed.get("final_score", 0))

    overall_summary = str(parsed.get("overall_summary", ""))


    return {

        "__status__": "ok",

        "valid": True,

        "tom1": tom1,

        "tom2": tom2,

        "strategy": strategy,

        "final_score": final_score,

        "overall_summary": overall_summary,

        "saved_at": int(time.time()),

    }


def judge_with_retry(client, prompt):

    last_error = None

    last_raw_text = ""

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            raw_text = call_judge_api(client, prompt)

            last_raw_text = raw_text or ""

            return parse_judge_response(raw_text)

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                time.sleep(1.5 * attempt)

    raise ValueError(f"Judge failed after retries: {last_error}; raw={last_raw_text[:600]}")


def record_to_result(record: Dict[str, Any], from_cache: bool) -> Dict[str, Any]:

    tom1 = record.get("tom1", {})

    tom2 = record.get("tom2", {})

    strategy = record.get("strategy", {})

    return {

        "valid": bool(record.get("valid", False)),

        "final_score": float(record.get("final_score", 0.0)),

        "tom1_score": float(tom1.get("score", 0.0)),

        "tom2_score": float(tom2.get("score", 0.0)),

        "strategy_score": float(strategy.get("score", 0.0)),

        "tom1_grounding": float(tom1.get("grounding", 0.0)),

        "tom1_coverage": float(tom1.get("coverage", 0.0)),

        "tom1_perspective_alignment": float(tom1.get("perspective_alignment", 0.0)),

        "tom1_conciseness": float(tom1.get("conciseness", 0.0)),

        "tom2_mental_modeling": float(tom2.get("mental_modeling", 0.0)),

        "tom2_consistency": float(tom2.get("consistency", 0.0)),

        "tom2_specificity": float(tom2.get("specificity", 0.0)),

        "tom2_risk_awareness": float(tom2.get("risk_awareness", 0.0)),

        "strategy_actionability": float(strategy.get("actionability", 0.0)),

        "strategy_coherence": float(strategy.get("coherence", 0.0)),

        "strategy_prioritization": float(strategy.get("prioritization", 0.0)),

        "strategy_risk_control": float(strategy.get("risk_control", 0.0)),

        "from_cache": from_cache,

        "error_counted": 1 if record.get("__status__") != "ok" else 0,

    }


def judge_one_sample(client, model_name, item_index, item, cache, cache_lock):

    output_text = item.get("model_output") or item.get("output", "")

    think = extract_think_json(output_text)

    answer_text = extract_answer(output_text)

    cache_key = make_cache_key(model_name, item, output_text)


    with cache_lock:

        cached = cache.get(cache_key)

    if cached is not None:

        return record_to_result(cached, from_cache=True)


    missing = [key for key in ("ToM1", "ToM2", "Strategy") if not str(think.get(key, "")).strip()]

    if missing:

        record = build_zero_record(f"think 缺少字段: {', '.join(missing)}", think)

        with cache_lock:

            cache[cache_key] = record

            save_cache(cache)

        return record_to_result(record, from_cache=False)


    prompt = build_judge_prompt(item, think, answer_text)

    try:

        record = judge_with_retry(client, prompt)

    except Exception as exc:

        append_error_log(

            {

                "model_name": model_name,

                "item_index": item_index,

                "task": "think_components",

                "error": str(exc),

                "input": item.get("input", "")[:2000],

                "model_output": output_text[:2000],

            }

        )

        record = build_zero_record(f"judge_error: {exc}", think)

        record["__status__"] = "judge_error"


    with cache_lock:

        cache[cache_key] = record

        save_cache(cache)

    return record_to_result(record, from_cache=False)


def summarize_model_results(model_name, results):

    total_samples = len(results)

    valid_count = 0

    final_score_all = 0.0

    tom1_score_all = 0.0

    tom2_score_all = 0.0

    strategy_score_all = 0.0

    tom1_grounding_all = 0.0

    tom1_coverage_all = 0.0

    tom1_perspective_all = 0.0

    tom1_conciseness_all = 0.0

    tom2_mental_all = 0.0

    tom2_consistency_all = 0.0

    tom2_specificity_all = 0.0

    tom2_risk_all = 0.0

    strategy_actionability_all = 0.0

    strategy_coherence_all = 0.0

    strategy_prioritization_all = 0.0

    strategy_risk_all = 0.0

    cache_hits = 0

    error_count = 0

    for result in results:

        valid_count += 1 if result["valid"] else 0

        cache_hits += result.get("from_cache", 0)

        error_count += result.get("error_counted", 0)

        final_score_all += result["final_score"]

        tom1_score_all += result["tom1_score"]

        tom2_score_all += result["tom2_score"]

        strategy_score_all += result["strategy_score"]

        tom1_grounding_all += result["tom1_grounding"]

        tom1_coverage_all += result["tom1_coverage"]

        tom1_perspective_all += result["tom1_perspective_alignment"]

        tom1_conciseness_all += result["tom1_conciseness"]

        tom2_mental_all += result["tom2_mental_modeling"]

        tom2_consistency_all += result["tom2_consistency"]

        tom2_specificity_all += result["tom2_specificity"]

        tom2_risk_all += result["tom2_risk_awareness"]

        strategy_actionability_all += result["strategy_actionability"]

        strategy_coherence_all += result["strategy_coherence"]

        strategy_prioritization_all += result["strategy_prioritization"]

        strategy_risk_all += result["strategy_risk_control"]

    denom = total_samples if total_samples else 1

    return {

        "model": model_name,

        "fmt_rate": (valid_count / total_samples) * 100 if total_samples else 0.0,

        "final_score": final_score_all / denom,

        "tom1_score": tom1_score_all / denom,

        "tom2_score": tom2_score_all / denom,

        "strategy_score": strategy_score_all / denom,

        "tom1_grounding": tom1_grounding_all / denom,

        "tom1_coverage": tom1_coverage_all / denom,

        "tom1_perspective_alignment": tom1_perspective_all / denom,

        "tom1_conciseness": tom1_conciseness_all / denom,

        "tom2_mental_modeling": tom2_mental_all / denom,

        "tom2_consistency": tom2_consistency_all / denom,

        "tom2_specificity": tom2_specificity_all / denom,

        "tom2_risk_awareness": tom2_risk_all / denom,

        "strategy_actionability": strategy_actionability_all / denom,

        "strategy_coherence": strategy_coherence_all / denom,

        "strategy_prioritization": strategy_prioritization_all / denom,

        "strategy_risk_control": strategy_risk_all / denom,

        "cache_hit_rate": (cache_hits / total_samples) * 100 if total_samples else 0.0,

        "error_count": error_count,

    }


def save_summary(summary_results, max_workers):

    payload = {

        "evaluator": EVAL_NAME,

        "saved_at": int(time.time()),

        "judge_model": JUDGE_MODEL,

        "use_llm": True,

        "max_workers": int(max_workers),

        "results": summary_results,

    }

    atomic_write_json(SUMMARY_FILE_PATH, payload)


def evaluate_model(client, model_name, data, cache, cache_lock, max_workers, result_path, partial_path, checkpoint_path):

    total_samples = len(data)


    existing_results = load_result_list(result_path)

    if existing_results is not None and len(existing_results) == total_samples:

        return summarize_model_results(model_name, existing_results)


    completed = load_checkpoint(checkpoint_path)

    if completed:

        atomic_write_json(partial_path, build_ordered_results(completed, total_samples))


    pending_indices = [idx for idx in range(total_samples) if idx not in completed]

    if pending_indices:

        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = {

                executor.submit(

                    judge_one_sample,

                    client,

                    model_name,

                    idx,

                    data[idx],

                    cache,

                    cache_lock,

                ): idx

                for idx in pending_indices

            }

            progress_bar = (

                tqdm(total=len(futures), desc=model_name, unit="sample") if tqdm is not None else None

            )

            try:

                for future in as_completed(futures):

                    idx = futures[future]

                    result = future.result()

                    completed[idx] = result

                    append_jsonl(checkpoint_path, {"index": idx, "result": result})

                    atomic_write_json(partial_path, build_ordered_results(completed, total_samples))

                    if progress_bar is not None:

                        progress_bar.update(1)

            finally:

                if progress_bar is not None:

                    progress_bar.close()


    ordered_results = build_ordered_results(completed, total_samples)

    if len(ordered_results) != total_samples:

        raise ValueError(

            f"{model_name} 评测结果不完整: {len(ordered_results)}/{total_samples}，请检查 {checkpoint_path}"

        )


    atomic_write_json(partial_path, ordered_results)

    atomic_write_json(result_path, ordered_results)

    return summarize_model_results(model_name, ordered_results)


def evaluate_think_components(args, verbose=False):

    if not JUDGE_API_KEY:

        raise ValueError("缺少 DEEPSEEK_API_KEY，无法调用 DeepSeek 判分。")

    if not os.path.exists(OUTPUT_BASE_DIR):

        print(f"❌ 找不到结果目录: {OUTPUT_BASE_DIR}")

        return


    client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)

    cache = load_cache()

    cache_lock = threading.Lock()


    print(f"已加载缓存 {len(cache)} 条 | 模式: Think 三段联合 LLM 评测 | 并发: {args.max_workers}")


    model_dirs = sorted(

        [d for d in os.listdir(OUTPUT_BASE_DIR) if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]

    )

    if SELECTED_MODEL_DIRS:

        model_dirs = [d for d in model_dirs if d in SELECTED_MODEL_DIRS]


    existing_model_dirs = [

        model_name

        for model_name in model_dirs

        if os.path.exists(os.path.join(OUTPUT_BASE_DIR, model_name, RESULT_FILE_NAME))

    ]


    summary_results = []

    total_models = len(existing_model_dirs)

    completed_models = 0

    overall_start_time = time.time()


    for model_name in existing_model_dirs:

        file_path = os.path.join(OUTPUT_BASE_DIR, model_name, RESULT_FILE_NAME)

        model_out_dir = os.path.join(OUTPUT_BASE_DIR, model_name)

        result_path = os.path.join(model_out_dir, EVAL_RESULT_FILE_NAME)

        partial_path = os.path.join(model_out_dir, EVAL_PARTIAL_FILE_NAME)

        checkpoint_path = os.path.join(model_out_dir, EVAL_CHECKPOINT_FILE_NAME)

        with open(file_path, "r", encoding="utf-8") as f:

            data = json.load(f)


        current_index = completed_models + 1

        elapsed_before = time.time() - overall_start_time

        avg_before = elapsed_before / completed_models if completed_models > 0 else 0.0

        remaining_before = avg_before * (total_models - completed_models) if completed_models > 0 else 0.0

        print(

            f"\n[总进度] {current_index}/{total_models} | 当前模型: {model_name} | "

            f"已耗时: {format_seconds(elapsed_before)} | 预计剩余: {format_seconds(remaining_before)}"

        )

        print(f"正在评测 Think 三段质量: {model_name} ({len(data)} samples)")

        if os.path.exists(checkpoint_path):

            completed_count = len(load_checkpoint(checkpoint_path))

            if completed_count > 0:

                print(f"♻️ [{model_name}] 检测到断点进度 {completed_count}/{len(data)}，将继续补跑。")


        model_start_time = time.time()

        model_summary = evaluate_model(

            client,

            model_name,

            data,

            cache,

            cache_lock,

            args.max_workers,

            result_path,

            partial_path,

            checkpoint_path,

        )

        summary_results.append(model_summary)

        save_summary(summary_results, args.max_workers)

        completed_models += 1

        model_elapsed = time.time() - model_start_time

        total_elapsed = time.time() - overall_start_time

        avg_after = total_elapsed / completed_models if completed_models > 0 else 0.0

        remaining_after = avg_after * (total_models - completed_models)

        print(

            f"[模型完成] {model_name} | 本模型耗时: {format_seconds(model_elapsed)} | "

            f"累计耗时: {format_seconds(total_elapsed)} | 预计剩余: {format_seconds(remaining_after)}"

        )


    if not summary_results:

        print("❌ 没有找到可评测结果。")

        return


    summary_results.sort(key=lambda x: x["final_score"], reverse=True)

    save_summary(summary_results, args.max_workers)


    def max_mark(value, max_value, width, suffix=""):

        base = f"{value:.1f}{suffix}"

        if value == max_value and max_value > 0:

            base = f"*{base}*"

        return base.ljust(width)


    max_final = max(r["final_score"] for r in summary_results)

    max_tom1 = max(r["tom1_score"] for r in summary_results)

    max_tom2 = max(r["tom2_score"] for r in summary_results)

    max_strategy = max(r["strategy_score"] for r in summary_results)

    max_cache = max(r["cache_hit_rate"] for r in summary_results)


    if verbose:

        widths = [24, 10, 10, 10, 10, 10]

        headers = ["Model Name", "Final", "ToM1", "ToM2", "Strategy", "Cache"]

        total_len = sum(widths) + len(widths) * 3 + 1

        print("\n" + "=" * total_len)

        print("📊 WereToM evaluation: explicit-ToM component quality")

        print("=" * total_len)

        print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")

        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")

        for r in summary_results:

            print(

                f"| {r['model'].ljust(widths[0])} | "

                f"{max_mark(r['final_score'], max_final, widths[1])} | "

                f"{max_mark(r['tom1_score'], max_tom1, widths[2])} | "

                f"{max_mark(r['tom2_score'], max_tom2, widths[3])} | "

                f"{max_mark(r['strategy_score'], max_strategy, widths[4])} | "

                f"{max_mark(r['cache_hit_rate'], max_cache, widths[5], '%')} |"

            )

            print(

                f"| {'details'.ljust(widths[0])} | "

                f"T1G:{r['tom1_grounding']:.1f} | T1C:{r['tom1_coverage']:.1f} | "

                f"T2M:{r['tom2_mental_modeling']:.1f} | SAct:{r['strategy_actionability']:.1f} | "

                f"err:{r['error_count']:<5} |"

            )

    else:

        widths = [24, 10, 10, 10, 10]

        headers = ["Model Name", "Final", "ToM1", "ToM2", "Strategy"]

        total_len = sum(widths) + len(widths) * 3 + 1

        print("\n" + "=" * total_len)

        print("📊 WereToM evaluation: explicit-ToM component quality")

        print("=" * total_len)

        print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")

        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")

        for r in summary_results:

            print(

                f"| {r['model'].ljust(widths[0])} | "

                f"{max_mark(r['final_score'], max_final, widths[1])} | "

                f"{max_mark(r['tom1_score'], max_tom1, widths[2])} | "

                f"{max_mark(r['tom2_score'], max_tom2, widths[3])} | "

                f"{max_mark(r['strategy_score'], max_strategy, widths[4])} |"

            )


    print("=" * total_len + "\n")


if __name__ == "__main__":

    args = parse_args()

    evaluate_think_components(args, verbose=resolve_verbose(args))
