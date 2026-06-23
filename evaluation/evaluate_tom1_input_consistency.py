import argparse

import hashlib

import json

import os

import re

import threading

import time

from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from typing import Any, Dict, List, Optional, Set, Tuple


from openai import OpenAI


try:

    from tqdm import tqdm

except ImportError:

    tqdm = None


OUTPUT_BASE_DIR = os.getenv("WERETOM_RESULT_DIR", "experiment_results")

RESULT_FILE_NAME = "results_vote_role_predict.json"

EVAL_NAME = "tom1_input_consistency"

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

LOCAL_WEIGHT = 0.3

LLM_WEIGHT = 0.7

CACHE_VERSION = "tom1_input_consistency_hybrid"

CACHE_PATH = Path(os.getenv("WERETOM_CACHE_DIR", OUTPUT_BASE_DIR)) / "tom1_input_consistency.cache.json"

ERROR_LOG_PATH = Path(os.getenv("WERETOM_CACHE_DIR", OUTPUT_BASE_DIR)) / "tom1_input_consistency.errors.jsonl"


ROLE_WORDS = {"平民", "狼人", "预言家", "女巫", "猎人"}

GOD_VIEW_MARKERS = {

    "真实身份", "上帝视角", "标准答案", "已知身份", "roles_map", "根据答案", "根据标注",

}


JUDGE_JSON_SYSTEM_PROMPT = (

    "You must output a valid json object only. "

    "Return strict JSON with double-quoted keys and string values where appropriate. "

    "Do not output markdown, code fences, explanations, or any text outside the json object. "

    "JSON output example: "

    "{\"grounding\": 88, \"coverage\": 84, \"faithfulness\": 91, "

    "\"viewpoint_alignment\": 86, \"total_score\": 87, \"summary\": \"example\"}."

)


def parse_args():

    parser = argparse.ArgumentParser(

        description="Evaluate ToM1-input consistency with hybrid local rules + DeepSeek judge."

    )

    parser.add_argument("--verbose", action="store_true", help="Print detailed sub-metrics.")

    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS, help="Thread count.")

    parser.add_argument("--no-llm", action="store_true", help="Use local regex scoring only.")

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


def extract_tom1(output_text):

    return extract_think_json(output_text).get("ToM1", "").strip()


def extract_answer(output_text):

    return extract_tag(output_text, "answer") or (output_text or "").strip()


def extract_alive_players(input_text: str) -> List[str]:

    match = re.search(r"【当前场上存活玩家】[：:]\s*\[(.*?)\]", input_text or "")

    return re.findall(r"\d+", match.group(1)) if match else []


def extract_self_seat(input_text: str) -> Optional[str]:

    match = re.search(r"【你的座位号】[：:]\s*\[(\d+)\]号", input_text or "")

    return match.group(1) if match else None


def extract_player_mentions(text: str) -> Set[str]:

    return set(re.findall(r"(\d+)\s*号", text or ""))


def extract_role_mentions(text: str) -> Set[str]:

    text = text or ""

    return {role for role in ROLE_WORDS if role in text}


def extract_bracket_sections(text: str) -> Set[str]:

    return set(re.findall(r"【([^】]+)】", text or ""))


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


def make_cache_key(model_name, item, output_text, tom1_text, use_llm):

    payload = {

        "version": CACHE_VERSION,

        "task": "tom1_input_consistency",

        "model_name": model_name,

        "input": item.get("input", ""),

        "player_id": item.get("player_id"),

        "player_role": item.get("player_role"),

        "output_text": output_text,

        "tom1": tom1_text,

        "judge_model": JUDGE_MODEL if use_llm else "local_only",

        "local_weight": LOCAL_WEIGHT,

        "llm_weight": LLM_WEIGHT,

    }

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def format_score_penalty(output_text: str) -> Tuple[float, List[str]]:

    reasons = []

    score = 1.0

    think = extract_think_json(output_text)

    answer = extract_answer(output_text)

    if not think:

        score -= 0.25

        reasons.append("think JSON 解析失败")

    for key in ("ToM1", "ToM2", "Strategy"):

        if key not in think or not str(think.get(key, "")).strip():

            score -= 0.08

            reasons.append(f"缺少 {key}")

    if not answer:

        score -= 0.20

        reasons.append("缺少 answer")

    return max(0.0, score), reasons


def score_local_tom1(item: Dict[str, Any], output_text: str) -> Dict[str, Any]:

    tom1_text = extract_tom1(output_text)

    factor, issues = format_score_penalty(output_text)

    if not tom1_text:

        return {

            "valid": False,

            "grounding": 0.0,

            "coverage": 0.0,

            "faithfulness": 0.0,

            "viewpoint_alignment": 0.0,

            "total_score": 0.0,

            "summary": "缺少 ToM1。",

            "details": {"issues": issues + ["缺少 ToM1"]},

        }


    input_text = item.get("input", "")

    self_seat = extract_self_seat(input_text)

    alive_players = set(extract_alive_players(input_text))

    input_seats = extract_player_mentions(input_text) | alive_players

    if self_seat:

        input_seats.add(self_seat)


    tom1_seats = extract_player_mentions(tom1_text)

    valid_tom1_seats = tom1_seats & input_seats

    hallucinated_seats = sorted(tom1_seats - input_seats, key=int)


    input_roles = extract_role_mentions(input_text)

    tom1_roles = extract_role_mentions(tom1_text)

    role_overlap = input_roles & tom1_roles


    input_sections = extract_bracket_sections(input_text)

    section_hit_count = sum(1 for section in input_sections if section and section in tom1_text)


    seat_precision = len(valid_tom1_seats) / len(tom1_seats) if tom1_seats else 0.75

    seat_coverage = len(valid_tom1_seats) / min(max(1, len(input_seats)), 4) if input_seats else 1.0

    role_precision = len(role_overlap) / len(tom1_roles) if tom1_roles else 0.70

    role_recall = len(role_overlap) / len(input_roles) if input_roles else 0.85

    section_recall = section_hit_count / len(input_sections) if input_sections else 0.75


    text_len = len(tom1_text)

    if text_len < 12:

        length_score = 0.40

    elif text_len <= 180:

        length_score = 1.0

    elif text_len <= 260:

        length_score = 0.80

    else:

        length_score = 0.60


    first_person_score = 1.0 if "我" in tom1_text else 0.55

    god_view_hits = sum(1 for marker in GOD_VIEW_MARKERS if marker in tom1_text)


    grounding = clamp(

        100 * (0.40 * seat_precision + 0.35 * role_precision + 0.25 * length_score)

        - 12 * len(hallucinated_seats)

        - 8 * god_view_hits

    )

    coverage = clamp(100 * (0.40 * seat_coverage + 0.35 * role_recall + 0.25 * section_recall))

    faithfulness = clamp(

        100

        - 20 * len(hallucinated_seats)

        - 18 * god_view_hits

        - (10 if "其实" in tom1_text and "我" not in tom1_text else 0)

    )

    viewpoint_alignment = clamp(

        100 * (0.50 * first_person_score + 0.25 * seat_precision + 0.25 * length_score)

        - 15 * god_view_hits

        - (8 if self_seat and self_seat in tom1_text and "我" not in tom1_text else 0)

    )


    base_total = 0.28 * grounding + 0.27 * coverage + 0.25 * faithfulness + 0.20 * viewpoint_alignment

    total_score = clamp(base_total * factor)


    summary_parts = []

    if hallucinated_seats:

        summary_parts.append(f"存在越界座位 {hallucinated_seats}")

    if seat_coverage < 0.30 and role_recall < 0.30:

        summary_parts.append("局势覆盖偏弱")

    if "我" not in tom1_text:

        summary_parts.append("第一视角较弱")

    if not summary_parts:

        summary_parts.append("本地规则判断基本一致")


    return {

        "valid": True,

        "grounding": round(grounding, 3),

        "coverage": round(coverage, 3),

        "faithfulness": round(faithfulness, 3),

        "viewpoint_alignment": round(viewpoint_alignment, 3),

        "total_score": round(total_score, 3),

        "summary": "；".join(summary_parts),

        "details": {

            "issues": issues,

            "hallucinated_seats": hallucinated_seats,

            "input_roles": sorted(input_roles),

            "tom1_roles": sorted(tom1_roles),

            "valid_tom1_seats": sorted(valid_tom1_seats, key=int),

            "god_view_hits": god_view_hits,

            "section_hit_count": section_hit_count,

        },

    }


def build_judge_prompt(item: Dict[str, Any], tom1_text: str, local: Dict[str, Any]) -> str:

    payload = {

        "player_id": item.get("player_id"),

        "player_role": item.get("player_role"),

        "input": item.get("input", ""),

        "tom1": tom1_text,

        "local_signals": {

            "grounding": local["grounding"],

            "coverage": local["coverage"],

            "faithfulness": local["faithfulness"],

            "viewpoint_alignment": local["viewpoint_alignment"],

            "total_score": local["total_score"],

            "details": local.get("details", {}),

        },

    }

    return (

        "你是狼人杀数据评测判官。现在要判断 ToM1 是否准确概括了 input。"

        "本地规则分只是辅助信号，你必须自己阅读 input 与 ToM1 后独立评分，不要机械照抄本地分。\n"

        "评分要求：\n"

        "1. 只判断 ToM1 与 input 是否一致，不要按真实身份结果打分。\n"

        "2. ToM1 是否抓住公开发言、票型、身份宣称、冲突点、局势走向。\n"

        "3. 若 ToM1 编造 input 中没有的信息，必须重罚。\n"

        "4. 若 ToM1 明显脱离当前玩家视角、缺少第一视角或带有上帝视角，要扣分。\n"

        "5. 输出严格 JSON，不要输出任何解释文字。\n\n"

        "返回字段：\n"

        "{\n"

        '  "grounding": 0-100,\n'

        '  "coverage": 0-100,\n'

        '  "faithfulness": 0-100,\n'

        '  "viewpoint_alignment": 0-100,\n'

        '  "total_score": 0-100,\n'

        '  "summary": "一句中文简评"\n'

        "}\n\n"

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

        max_tokens=700,

        timeout=REQUEST_TIMEOUT,

        response_format={"type": "json_object"},

    )

    return response.choices[0].message.content or ""


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

    return {

        "grounding": clamp(parsed.get("grounding", 0)),

        "coverage": clamp(parsed.get("coverage", 0)),

        "faithfulness": clamp(parsed.get("faithfulness", 0)),

        "viewpoint_alignment": clamp(parsed.get("viewpoint_alignment", 0)),

        "total_score": clamp(parsed.get("total_score", 0)),

        "summary": str(parsed.get("summary", "")),

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

    raise ValueError(f"Judge failed after retries: {last_error}; raw={last_raw_text[:500]}")


def blend_scores(local: Dict[str, Any], llm: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    if not llm:

        return {

            "grounding": round(local["grounding"], 3),

            "coverage": round(local["coverage"], 3),

            "faithfulness": round(local["faithfulness"], 3),

            "viewpoint_alignment": round(local["viewpoint_alignment"], 3),

            "total_score": round(local["total_score"], 3),

        }

    return {

        "grounding": round(LOCAL_WEIGHT * local["grounding"] + LLM_WEIGHT * llm["grounding"], 3),

        "coverage": round(LOCAL_WEIGHT * local["coverage"] + LLM_WEIGHT * llm["coverage"], 3),

        "faithfulness": round(LOCAL_WEIGHT * local["faithfulness"] + LLM_WEIGHT * llm["faithfulness"], 3),

        "viewpoint_alignment": round(

            LOCAL_WEIGHT * local["viewpoint_alignment"] + LLM_WEIGHT * llm["viewpoint_alignment"], 3

        ),

        "total_score": round(LOCAL_WEIGHT * local["total_score"] + LLM_WEIGHT * llm["total_score"], 3),

    }


def make_record(local: Dict[str, Any], llm: Optional[Dict[str, Any]], mode: str, valid: bool, error: str = "") -> Dict[str, Any]:

    hybrid = blend_scores(local, llm)

    return {

        "__status__": "ok" if not error else "fallback_local",

        "mode": mode,

        "valid": valid,

        "local": local,

        "llm": llm,

        "hybrid": hybrid,

        "final_score": hybrid["total_score"],

        "error": error,

        "saved_at": int(time.time()),

    }


def record_to_result(record: Dict[str, Any], from_cache: bool) -> Dict[str, Any]:

    hybrid = record.get("hybrid", {})

    local = record.get("local", {})

    llm = record.get("llm") or {}

    return {

        "valid": bool(record.get("valid", False)),

        "score": float(record.get("final_score", hybrid.get("total_score", 0.0))),

        "grounding": float(hybrid.get("grounding", 0.0)),

        "coverage": float(hybrid.get("coverage", 0.0)),

        "faithfulness": float(hybrid.get("faithfulness", 0.0)),

        "viewpoint_alignment": float(hybrid.get("viewpoint_alignment", 0.0)),

        "local_score": float(local.get("total_score", 0.0)),

        "llm_score": float(llm.get("total_score", 0.0)) if llm else 0.0,

        "llm_used": 1 if llm else 0,

        "from_cache": from_cache,

        "fallback_local": 1 if record.get("mode") == "fallback_local" else 0,

    }


def judge_one_sample(client, model_name, item_index, item, cache, cache_lock, use_llm):

    output_text = item.get("model_output") or item.get("output", "")

    tom1_text = extract_tom1(output_text)

    cache_key = make_cache_key(model_name, item, output_text, tom1_text, use_llm)


    with cache_lock:

        cached = cache.get(cache_key)

    if cached is not None:

        return record_to_result(cached, from_cache=True)


    local = score_local_tom1(item, output_text)

    if not local["valid"]:

        record = make_record(local, None, mode="local_invalid", valid=False)

        with cache_lock:

            cache[cache_key] = record

            save_cache(cache)

        return record_to_result(record, from_cache=False)


    if not use_llm or client is None:

        record = make_record(local, None, mode="local_only", valid=True)

        with cache_lock:

            cache[cache_key] = record

            save_cache(cache)

        return record_to_result(record, from_cache=False)


    prompt = build_judge_prompt(item, tom1_text, local)

    try:

        llm = judge_with_retry(client, prompt)

        record = make_record(local, llm, mode="hybrid", valid=True)

    except Exception as exc:

        append_error_log(

            {

                "model_name": model_name,

                "item_index": item_index,

                "task": "tom1",

                "error": str(exc),

                "input": item.get("input", "")[:2000],

                "model_output": output_text[:2000],

            }

        )

        record = make_record(local, None, mode="fallback_local", valid=True, error=str(exc))


    with cache_lock:

        cache[cache_key] = record

        save_cache(cache)

    return record_to_result(record, from_cache=False)


def summarize_model_results(model_name, results):

    total_samples = len(results)

    valid_count = 0

    score_all = 0.0

    score_valid = 0.0

    local_score_all = 0.0

    llm_score_all = 0.0

    llm_count = 0

    grounding_all = 0.0

    coverage_all = 0.0

    faithfulness_all = 0.0

    viewpoint_all = 0.0

    cache_hits = 0

    fallback_local = 0

    for result in results:

        cache_hits += result.get("from_cache", 0)

        fallback_local += result.get("fallback_local", 0)

        if result["valid"]:

            valid_count += 1

            score_valid += result["score"]

        score_all += result["score"]

        local_score_all += result["local_score"]

        grounding_all += result["grounding"]

        coverage_all += result["coverage"]

        faithfulness_all += result["faithfulness"]

        viewpoint_all += result["viewpoint_alignment"]

        if result["llm_used"]:

            llm_count += 1

            llm_score_all += result["llm_score"]

    return {

        "model": model_name,

        "fmt_rate": (valid_count / total_samples) * 100 if total_samples else 0.0,

        "final_score": score_all / total_samples if total_samples else 0.0,

        "score_valid": score_valid / valid_count if valid_count else 0.0,

        "score_all": score_all / total_samples if total_samples else 0.0,

        "local_score": local_score_all / total_samples if total_samples else 0.0,

        "llm_score": llm_score_all / llm_count if llm_count else 0.0,

        "grounding": grounding_all / total_samples if total_samples else 0.0,

        "coverage": coverage_all / total_samples if total_samples else 0.0,

        "faithfulness": faithfulness_all / total_samples if total_samples else 0.0,

        "viewpoint_alignment": viewpoint_all / total_samples if total_samples else 0.0,

        "cache_hit_rate": (cache_hits / total_samples) * 100 if total_samples else 0.0,

        "fallback_local": fallback_local,

    }


def save_summary(summary_results, use_llm, max_workers):

    payload = {

        "evaluator": EVAL_NAME,

        "saved_at": int(time.time()),

        "judge_model": JUDGE_MODEL if use_llm else "local_only",

        "use_llm": bool(use_llm),

        "max_workers": int(max_workers),

        "results": summary_results,

    }

    atomic_write_json(SUMMARY_FILE_PATH, payload)


def evaluate_model(

    client,

    model_name,

    data,

    cache,

    cache_lock,

    max_workers,

    use_llm,

    result_path,

    partial_path,

    checkpoint_path,

):

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

                    use_llm,

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


def evaluate_tom1_consistency(args, verbose=False):

    if not os.path.exists(OUTPUT_BASE_DIR):

        print(f"❌ 找不到结果目录: {OUTPUT_BASE_DIR}")

        return


    use_llm = (not args.no_llm) and bool(JUDGE_API_KEY) and bool(JUDGE_MODEL)

    client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY) if use_llm else None

    cache = load_cache()

    cache_lock = threading.Lock()


    print(

        f"已加载缓存 {len(cache)} 条 | 模式: {'本地+LLM混合(30/70)' if use_llm else '仅本地规则'} | 并发: {args.max_workers}"

    )


    model_dirs = sorted([d for d in os.listdir(OUTPUT_BASE_DIR) if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))])

    if SELECTED_MODEL_DIRS:

        model_dirs = [d for d in model_dirs if d in SELECTED_MODEL_DIRS]


    existing_model_dirs = [

        model_name for model_name in model_dirs

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

        print(f"正在评测 ToM1 一致性: {model_name} ({len(data)} samples)")

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

            use_llm,

            result_path,

            partial_path,

            checkpoint_path,

        )

        summary_results.append(model_summary)

        save_summary(summary_results, use_llm, args.max_workers)

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

    save_summary(summary_results, use_llm, args.max_workers)


    max_final = max(r["final_score"] for r in summary_results)

    max_local = max(r["local_score"] for r in summary_results)

    max_llm = max(r["llm_score"] for r in summary_results)

    max_grounding = max(r["grounding"] for r in summary_results)

    max_coverage = max(r["coverage"] for r in summary_results)

    max_faithfulness = max(r["faithfulness"] for r in summary_results)

    max_viewpoint = max(r["viewpoint_alignment"] for r in summary_results)

    max_cache = max(r["cache_hit_rate"] for r in summary_results)


    def max_mark(value, max_value, width, suffix=""):

        base = f"{value:.1f}{suffix}"

        if value == max_value and max_value > 0:

            base = f"*{base}*"

        return base.ljust(width)


    if verbose:

        widths = [24, 10, 10, 10, 10, 10, 10, 10, 10]

        headers = ["Model Name", "Final", "Local", "LLM", "Ground", "Cover", "Faith", "View", "Cache"]

        total_len = sum(widths) + len(widths) * 3 + 1

        print("\n" + "=" * total_len)

        print("📊 WereToM evaluation: ToM1-input consistency")

        print("=" * total_len)

        print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")

        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")


        for r in summary_results:

            row = (

                f"| {r['model'].ljust(widths[0])} | "

                f"{max_mark(r['final_score'], max_final, widths[1])} | "

                f"{max_mark(r['local_score'], max_local, widths[2])} | "

                f"{max_mark(r['llm_score'], max_llm, widths[3])} | "

                f"{max_mark(r['grounding'], max_grounding, widths[4])} | "

                f"{max_mark(r['coverage'], max_coverage, widths[5])} | "

                f"{max_mark(r['faithfulness'], max_faithfulness, widths[6])} | "

                f"{max_mark(r['viewpoint_alignment'], max_viewpoint, widths[7])} | "

                f"{max_mark(r['cache_hit_rate'], max_cache, widths[8], '%')} |"

            )

            print(row)

    else:

        widths = [28, 12]

        headers = ["Model Name", "Final"]

        total_len = sum(widths) + len(widths) * 3 + 1

        print("\n" + "=" * total_len)

        print("📊 WereToM evaluation: ToM1-input consistency")

        print("=" * total_len)

        print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")

        print("|" + "|".join("-" * (w + 2) for w in widths) + "|")

        for r in summary_results:

            print(f"| {r['model'].ljust(widths[0])} | {max_mark(r['final_score'], max_final, widths[1])} |")


    print("=" * total_len + "\n")


if __name__ == "__main__":

    args = parse_args()

    evaluate_tom1_consistency(args, verbose=resolve_verbose(args))
