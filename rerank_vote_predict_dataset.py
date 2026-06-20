import argparse
import hashlib
import json
import random
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


# ============================================================
# 配置区
#
# 使用方式：
# 1. 常规情况下只需要改本区域，或在命令行里传同名参数覆盖。
# 2. API key 直接复制到下面的 GENERATOR_API_KEY / JUDGE_API_KEY 字符串里。
# 3. JUDGE_MODEL 默认为空，表示暂不启用 LLM 判官，只使用本地启发式评分。
# 4. 跑全量前建议先用 --dry-run、--limit 5 或 --sample 20 小规模检查输出。
# ============================================================

# 路径配置：
# - 脚本和所有新生成文件默认放在 wolf_test/rerank_vote_predict。
# - 原始训练集仍从 wolf_test/data 读取。
# - 如需临时输出到其他位置，推荐用命令行参数 --output/--checkpoint/--error-log 覆盖。
OUTPUT_DIR = Path(__file__).resolve().parent
WOLF_TEST_DIR = OUTPUT_DIR.parent
DATA_DIR = WOLF_TEST_DIR / "data"

# 原始数据集路径：一般不需要改，除非你换了输入数据版本。
INPUT_FILE = DATA_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_end.json"

# 最终重排后的数据集路径：默认只保留原始字段，不把评分元数据写进训练样本。
OUTPUT_FILE = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_reranked.json"

# 断点续传文件：每处理完一条就追加一行 JSONL。中断后重跑会跳过已完成 index。
CHECKPOINT_FILE = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_rerank.checkpoint.jsonl"

# 错误日志：API 调用、候选解析或判官解析失败时记录到这里，方便事后排查。
ERROR_LOG_FILE = OUTPUT_DIR / "werewolf_train_deepseek_clean_v4_vote_predict_only_rerank.errors.jsonl"

# 生成模型配置：
# - 用于对每条样本重新生成 N_CANDIDATES 个候选 output。
# - 需要服务兼容 OpenAI Chat Completions 接口。
# - 如果你的 DeepSeek 模型名后续变化，只改 GENERATOR_MODEL 即可。
GENERATOR_BASE_URL = "https://api.deepseek.com"
GENERATOR_API_KEY = "sk-35ee3b32f2d34547bd9355c52240ddae"  # TODO: 在这里粘贴 DeepSeek API key，例如 "sk-..."
GENERATOR_MODEL = "deepseek-chat"

# 判官模型配置：
# - 用于综合判断“结果准确性”和“一致性”。
# - 你还没确定模型时，让 JUDGE_MODEL 保持 ""，不会产生判官模型费用。
# - 确定后可改为目标模型名，例如 "gpt-5.4"，或命令行传 --judge-model gpt-5.4。
# - JUDGE_BASE_URL 默认是 OpenAI；如果使用其他兼容接口，改成对应 base_url。
JUDGE_BASE_URL = "https://api.mirrorworkforce.cn/v1"
JUDGE_API_KEY = "sk-cYK60vDu8it5VGJ3K7Y4CSk1AhruYcBShUYT2tPalWMz3w4I"  # TODO: 在这里粘贴判官模型 API key，例如 "sk-..."
JUDGE_MODEL = "gpt-5.4"  # 留空则不启用判官模型
ENABLE_LLM_JUDGE = bool(JUDGE_API_KEY and JUDGE_MODEL)

# 运行规模与稳定性：
# - N_CANDIDATES：每条样本新生成多少个候选；你的需求是 3。
# - MAX_RETRIES：API 失败重试次数。
# - REQUEST_TIMEOUT：单次 API 请求超时时间，单位秒。
# - MAX_WORKERS：样本级并发数。每个样本会触发 3 次生成 + 4 次判官；太高容易 429。
# - STRICT_API_FAILURE：正式全量建议保持 True；生成或判官 API 失败时停止，避免混入本地兜底评分结果。
# - SKIP_SENSITIVE_JUDGE_BLOCK：判官通道误拦狼人杀术语时跳过该样本，避免卡住全量任务。
# - SAVE_EVERY：每处理多少条在控制台打印一次 token 用量统计。
# - PRINT_EVERY_ITEM：是否每条样本开始/完成都打印进度；100条试跑建议保持 True。
# - SLEEP_BETWEEN_ITEMS：需要限速时可设为 0.2、1.0 等秒数。
N_CANDIDATES = 3
MAX_RETRIES = 3
REQUEST_TIMEOUT = 120
MAX_WORKERS = 3
STRICT_API_FAILURE = True
SKIP_SENSITIVE_JUDGE_BLOCK = True
SAVE_EVERY = 20
PRINT_EVERY_ITEM = True
SLEEP_BETWEEN_ITEMS = 0.0

# 综合评分权重：
# - 启用 LLM 判官时：final = LOCAL_SCORE_WEIGHT * 本地分 + LLM_SCORE_WEIGHT * 判官分。
# - 未启用 LLM 判官时：只使用本地分，这两个权重不会影响结果。
# - 如果你更信任判官模型，可以提高 LLM_SCORE_WEIGHT，例如 0.8。
LOCAL_SCORE_WEIGHT = 0.3
LLM_SCORE_WEIGHT = 0.7

# roles_map 英文身份到 answer 中文身份标签的映射。
# 数据集 answer 要求只能使用：平民、狼人、预言家、女巫、猎人。
ROLE_EN_TO_ZH = {
    "Villager": "平民",
    "Werewolf": "狼人",
    "Seer": "预言家",
    "Witch": "女巫",
    "Hunter": "猎人",
}
VALID_ZH_ROLES = set(ROLE_EN_TO_ZH.values())


class SensitiveJudgeBlock(RuntimeError):
    """Raised when the judge API blocks a werewolf-game sample as sensitive."""


def is_sensitive_judge_error(exc: Exception) -> bool:
    return "sensitive_words_detected" in repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 3 new answers for each werewolf vote-prediction sample, judge them, and keep the best."
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_FILE)
    parser.add_argument("--error-log", type=Path, default=ERROR_LOG_FILE)
    parser.add_argument("--start", type=int, default=0, help="0-based start index.")
    parser.add_argument("--limit", type=int, default=None, help="Small experiment size.")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N items after --start/--limit filtering.")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Sample-level concurrency.")
    parser.add_argument("--generator-model", default=GENERATOR_MODEL)
    parser.add_argument("--generator-base-url", default=GENERATOR_BASE_URL)
    parser.add_argument("--generator-api-key", default=GENERATOR_API_KEY)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    parser.add_argument("--no-llm-judge", action="store_true", help="Only use local heuristic scoring.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call APIs; score only the original output.")
    parser.add_argument(
        "--allow-local-fallback",
        action="store_true",
        help="Allow local heuristic fallback when the LLM judge fails. Not recommended for final full runs.",
    )
    parser.add_argument("--keep-meta", action="store_true", help="Keep _rerank_meta in final output records.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=700)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_hash(item: Dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
            "roles_map": item.get("roles_map", {}),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_tag(text: str, tag: str) -> str:
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_think_json(output_text: str) -> Dict[str, str]:
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


def extract_answer(output_text: str) -> str:
    return extract_tag(output_text, "answer") or (output_text or "").strip()


def extract_vote_target(output_text: str) -> Optional[str]:
    answer = extract_answer(output_text)
    patterns = [
        r"投票(?:给|给了|目标为)?\s*【?\s*(\d+)\s*号?】?",
        r"【\s*(\d+)\s*号?\s*】",
        r"票(?:给|挂|投)\s*(\d+)\s*号?",
    ]
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(1)
    return None


def extract_prediction_json(output_text: str) -> Dict[str, str]:
    answer = extract_answer(output_text)
    matches = re.findall(r"\{.*?\}", answer, re.DOTALL)
    for raw in reversed(matches):
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return {str(k): str(v).strip() for k, v in parsed.items()}
    return {}


def extract_alive_players(input_text: str) -> List[str]:
    match = re.search(r"【当前场上存活玩家】[：:]\s*\[(.*?)\]", input_text or "")
    return re.findall(r"\d+", match.group(1)) if match else []


def extract_self_seat(input_text: str) -> Optional[str]:
    match = re.search(r"【你的座位号】[：:]\s*\[(\d+)\]号", input_text or "")
    return match.group(1) if match else None


def zh_roles_map(roles_map: Dict[str, str]) -> Dict[str, str]:
    return {str(k): ROLE_EN_TO_ZH.get(str(v), str(v)) for k, v in roles_map.items()}


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
        score -= 0.2
        reasons.append("缺少 answer")
    if extract_vote_target(output_text) is None:
        score -= 0.15
        reasons.append("投票目标解析失败")
    if not extract_prediction_json(output_text):
        score -= 0.15
        reasons.append("身份预测 JSON 解析失败")
    return max(0.0, score), reasons


def local_accuracy_score(item: Dict[str, Any], output_text: str) -> Tuple[float, Dict[str, Any]]:
    roles = zh_roles_map(item.get("roles_map", {}))
    alive = set(extract_alive_players(item.get("input", "")))
    self_seat = extract_self_seat(item.get("input", ""))
    predictions = extract_prediction_json(output_text)
    vote_target = extract_vote_target(output_text)

    target_players = sorted((alive or set(roles)) - ({self_seat} if self_seat else set()), key=int)
    pred_total = len(target_players)
    pred_correct = 0
    pred_valid = 0
    for seat in target_players:
        predicted = predictions.get(seat)
        if predicted in VALID_ZH_ROLES:
            pred_valid += 1
        if predicted == roles.get(seat):
            pred_correct += 1

    pred_score = pred_correct / pred_total if pred_total else 0.0
    coverage_score = pred_valid / pred_total if pred_total else 0.0

    self_role = item.get("roles_map", {}).get(str(self_seat), "")
    target_role = item.get("roles_map", {}).get(str(vote_target), "")
    vote_score = 0.4
    if vote_target and vote_target in alive and vote_target != self_seat:
        if self_role == "Werewolf":
            vote_score = 1.0 if target_role != "Werewolf" else 0.2
        else:
            vote_score = 1.0 if target_role == "Werewolf" else 0.35

    score = 0.55 * pred_score + 0.25 * vote_score + 0.20 * coverage_score
    details = {
        "prediction_correct": pred_correct,
        "prediction_total": pred_total,
        "prediction_valid": pred_valid,
        "vote_target": vote_target,
        "vote_target_role": target_role,
        "self_role": self_role,
    }
    return round(score * 100, 3), details


def local_consistency_score(item: Dict[str, Any], output_text: str) -> Tuple[float, Dict[str, Any]]:
    think = extract_think_json(output_text)
    answer = extract_answer(output_text)
    vote_target = extract_vote_target(output_text)
    predictions = extract_prediction_json(output_text)
    alive = set(extract_alive_players(item.get("input", "")))
    self_seat = extract_self_seat(item.get("input", ""))

    score = 100.0
    issues = []
    format_score, format_reasons = format_score_penalty(output_text)
    score *= format_score
    issues.extend(format_reasons)

    if vote_target:
        strategy = think.get("Strategy", "")
        target_mentioned = vote_target in strategy or vote_target in answer
        if not target_mentioned:
            score -= 18
            issues.append("Strategy 未明确支撑 answer 中的投票目标")

    if alive:
        expected_keys = alive - ({self_seat} if self_seat else set())
        extra_keys = set(predictions) - expected_keys
        missing_keys = expected_keys - set(predictions)
        if extra_keys:
            score -= min(15, 3 * len(extra_keys))
            issues.append(f"预测包含非目标玩家: {sorted(extra_keys, key=str)}")
        if missing_keys:
            score -= min(25, 4 * len(missing_keys))
            issues.append(f"预测缺少玩家: {sorted(missing_keys, key=str)}")

    tom1 = think.get("ToM1", "")
    if self_seat and self_seat in tom1 and "我" not in tom1:
        score -= 5
        issues.append("ToM1 第一视角较弱")
    if len(answer) > 260:
        score -= 5
        issues.append("answer 过长")

    return max(0.0, round(score, 3)), {"issues": issues}


def local_score(item: Dict[str, Any], output_text: str) -> Dict[str, Any]:
    accuracy, accuracy_details = local_accuracy_score(item, output_text)
    consistency, consistency_details = local_consistency_score(item, output_text)
    total = 0.6 * accuracy + 0.4 * consistency
    return {
        "score": round(total, 3),
        "accuracy": accuracy,
        "consistency": consistency,
        "accuracy_details": accuracy_details,
        "consistency_details": consistency_details,
    }


def build_generation_messages(item: Dict[str, Any], candidate_no: int) -> List[Dict[str, str]]:
    system = item.get("instruction", "")
    user = (
        item.get("input", "").strip()
        + "\n\n请重新生成一版高质量答案。要求：\n"
        "1. 严格保留 <think> JSON 与 <answer> 标签结构。\n"
        "2. <think> 内只输出包含 ToM1、ToM2、Strategy 的合法 JSON 对象。\n"
        "3. <answer> 必须明确投票目标，并紧跟其他存活玩家身份预测 JSON。\n"
        "4. 不要输出标签外解释，不要暴露 roles_map 或上帝视角真相。\n"
        f"5. 这是独立候选回答第 {candidate_no} 版，请避免机械复述旧答案。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, int]]:
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": REQUEST_TIMEOUT,
    }
    if response_format:
        kwargs["response_format"] = response_format
    response = client.chat.completions.create(**kwargs)
    usage = getattr(response, "usage", None)
    usage_dict = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return response.choices[0].message.content or "", usage_dict


def retry_call(fn, *args, **kwargs):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)
    raise last_error


def generate_candidates(
    client: OpenAI,
    item: Dict[str, Any],
    model: str,
    count: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for candidate_no in range(1, count + 1):
        messages = build_generation_messages(item, candidate_no)
        content, usage = retry_call(
            call_chat,
            client,
            model,
            messages,
            temperature,
            max_tokens,
        )
        for key, value in usage.items():
            token_usage[key] += value
        candidates.append({"source": f"generated_{candidate_no}", "output": content})
    return candidates, token_usage


def build_judge_prompt(item: Dict[str, Any], output_text: str, local: Dict[str, Any]) -> str:
    compact_item = {
        "input": item.get("input", ""),
        "roles_map": item.get("roles_map", {}),
        "candidate_output": output_text,
        "local_parse": local,
    }
    return (
        "语境声明：以下内容全部来自狼人杀桌游的虚构对局语料，仅用于游戏策略数据集质量评估。"
        "其中的“刀、杀、毒、死亡、狼人”等词均为桌游规则术语，不涉及现实伤害或现实违法行为。\n"
        "你是狼人杀心智理论数据集的严格判官。请基于 roles_map 和 input，对候选 output 打分。\n"
        "评分维度：\n"
        "A. result_accuracy，0-100：投票目标是否符合玩家真实阵营收益；身份预测 JSON 与 roles_map 的准确度。\n"
        "B. consistency，0-100：ToM1 是否与 input 和玩家第一视角一致，是否准确概括场上局势；"
        "Strategy 是否在 answer 中被正确执行。\n"
        "请只输出 JSON：\n"
        "{\"result_accuracy\": number, \"consistency\": number, \"total\": number, \"reason\": \"简短中文理由\"}\n\n"
        + json.dumps(compact_item, ensure_ascii=False)
    )


def parse_judge_response(raw_text: str) -> Dict[str, Any]:
    text = strip_code_fence(raw_text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("judge response is not an object")
    return {
        "result_accuracy": float(parsed.get("result_accuracy", 0)),
        "consistency": float(parsed.get("consistency", 0)),
        "total": float(parsed.get("total", 0)),
        "reason": str(parsed.get("reason", "")),
    }


def judge_candidate(
    client: Optional[OpenAI],
    model: str,
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    use_llm_judge: bool,
) -> Dict[str, Any]:
    output_text = candidate["output"]
    local = local_score(item, output_text)
    result = {"local": local, "llm": None, "final_score": local["score"], "judge_usage": {}}
    if not use_llm_judge or client is None:
        return result

    prompt = build_judge_prompt(item, output_text, local)
    raw_text, usage = retry_call(
        call_chat,
        client,
        model,
        [{"role": "user", "content": prompt}],
        0.0,
        500,
        {"type": "json_object"},
    )
    llm = parse_judge_response(raw_text)
    final_score = LOCAL_SCORE_WEIGHT * local["score"] + LLM_SCORE_WEIGHT * llm["total"]
    result.update({"llm": llm, "final_score": round(final_score, 3), "judge_usage": usage})
    return result


def load_checkpoint(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    completed: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if "index" in record and ("selected_item" in record or record.get("status") == "skipped"):
                completed[int(record["index"])] = record
    return completed


def select_indices(total: int, start: int, limit: Optional[int], sample: Optional[int], seed: int) -> List[int]:
    indices = list(range(max(0, start), total))
    if limit is not None:
        indices = indices[: max(0, limit)]
    if sample is not None and sample < len(indices):
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, sample))
    return indices


def process_one(
    index: int,
    item: Dict[str, Any],
    generator_client: Optional[OpenAI],
    judge_client: Optional[OpenAI],
    args: argparse.Namespace,
    use_llm_judge: bool,
) -> Dict[str, Any]:
    original_candidate = {"source": "original", "output": item.get("output", "")}
    generation_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    candidates = [original_candidate]

    if not args.dry_run:
        if generator_client is None:
            raise RuntimeError("缺少生成模型 API key。请在配置区填写 GENERATOR_API_KEY。")
        generated, generation_usage = generate_candidates(
            generator_client,
            item,
            args.generator_model,
            args.candidates,
            args.temperature,
            args.max_tokens,
        )
        candidates.extend(generated)

    scored_candidates = []
    judge_usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for candidate in candidates:
        try:
            judgement = judge_candidate(judge_client, args.judge_model, item, candidate, use_llm_judge)
        except Exception as exc:
            append_jsonl(
                args.error_log,
                {
                    "index": index,
                    "item_hash": stable_hash(item),
                    "stage": "judge",
                    "candidate_source": candidate["source"],
                    "error": repr(exc),
                },
            )
            if SKIP_SENSITIVE_JUDGE_BLOCK and is_sensitive_judge_error(exc):
                raise SensitiveJudgeBlock(
                    f"判官通道敏感词误拦截，跳过该样本。index={index}, "
                    f"candidate={candidate['source']}, error={exc!r}"
                ) from exc
            if STRICT_API_FAILURE and not args.allow_local_fallback:
                raise RuntimeError(
                    f"判官模型调用失败，严格模式已停止。index={index}, "
                    f"candidate={candidate['source']}, error={exc!r}"
                ) from exc
            judgement = {
                "local": local_score(item, candidate["output"]),
                "llm": None,
                "final_score": local_score(item, candidate["output"])["score"],
                "judge_usage": {},
                "judge_error": repr(exc),
            }
        for key, value in judgement.get("judge_usage", {}).items():
            judge_usage_total[key] += value
        scored_candidates.append({**candidate, "judgement": judgement})

    original_score = scored_candidates[0]["judgement"]["final_score"]
    generated_best = max(scored_candidates[1:], key=lambda c: c["judgement"]["final_score"], default=None)
    if generated_best and generated_best["judgement"]["final_score"] > original_score:
        selected = generated_best
    else:
        selected = scored_candidates[0]

    selected_item = dict(item)
    selected_item["output"] = selected["output"]
    if args.keep_meta:
        selected_item["_rerank_meta"] = {
            "selected_source": selected["source"],
            "selected_score": selected["judgement"]["final_score"],
            "original_score": original_score,
            "item_hash": stable_hash(item),
        }

    return {
        "index": index,
        "item_hash": stable_hash(item),
        "selected_source": selected["source"],
        "selected_item": selected_item,
        "scores": [
            {
                "source": candidate["source"],
                "final_score": candidate["judgement"]["final_score"],
                "local": candidate["judgement"]["local"],
                "llm": candidate["judgement"].get("llm"),
                "judge_error": candidate["judgement"].get("judge_error"),
            }
            for candidate in scored_candidates
        ],
        "usage": {"generation": generation_usage, "judge": judge_usage_total},
    }


def process_one_with_clients(
    index: int,
    item: Dict[str, Any],
    args: argparse.Namespace,
    use_llm_judge: bool,
) -> Dict[str, Any]:
    generator_client = None
    judge_client = None
    if not args.dry_run:
        generator_client = OpenAI(api_key=args.generator_api_key, base_url=args.generator_base_url)
    if use_llm_judge:
        judge_client = OpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    return process_one(index, item, generator_client, judge_client, args, use_llm_judge)


def summarize_usage(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    total = {
        "generation_prompt_tokens": 0,
        "generation_completion_tokens": 0,
        "generation_total_tokens": 0,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_total_tokens": 0,
    }
    for record in records:
        usage = record.get("usage", {})
        generation = usage.get("generation", {})
        judge = usage.get("judge", {})
        total["generation_prompt_tokens"] += int(generation.get("prompt_tokens", 0) or 0)
        total["generation_completion_tokens"] += int(generation.get("completion_tokens", 0) or 0)
        total["generation_total_tokens"] += int(generation.get("total_tokens", 0) or 0)
        total["judge_prompt_tokens"] += int(judge.get("prompt_tokens", 0) or 0)
        total["judge_completion_tokens"] += int(judge.get("completion_tokens", 0) or 0)
        total["judge_total_tokens"] += int(judge.get("total_tokens", 0) or 0)
    return total


def main() -> None:
    args = parse_args()
    data = read_json(args.input)
    if not isinstance(data, list):
        raise TypeError(f"dataset top-level must be list, got {type(data).__name__}")

    use_llm_judge = bool(args.judge_api_key and args.judge_model and not args.no_llm_judge)
    completed = load_checkpoint(args.checkpoint)
    selected_indices = select_indices(len(data), args.start, args.limit, args.sample, args.seed)
    selected_set = set(selected_indices)
    workers = max(1, int(args.workers or 1))

    print(f"dataset size: {len(data)}")
    print(f"target items this run: {len(selected_indices)}")
    print(f"checkpoint loaded: {len(completed)}")
    print(f"llm judge enabled: {use_llm_judge}")
    print(f"dry run: {args.dry_run}")
    print(f"workers: {workers}")

    run_records: List[Dict[str, Any]] = []
    pending: List[Tuple[int, int]] = []
    for offset, index in enumerate(selected_indices, start=1):
        if index in completed:
            if PRINT_EVERY_ITEM:
                print(f"[{offset}/{len(selected_indices)}] skip index={index}: checkpoint exists", flush=True)
            continue
        pending.append((offset, index))

    def submit_one(executor, offset: int, index: int):
        item = data[index]
        if PRINT_EVERY_ITEM:
            print(
                f"[{offset}/{len(selected_indices)}] start index={index}: "
                f"generate {args.candidates} candidates, judge={'on' if use_llm_judge else 'off'}",
                flush=True,
            )
        future = executor.submit(process_one_with_clients, index, item, args, use_llm_judge)
        return future, (offset, index, item)

    done_count = len(completed)
    pending_cursor = 0
    first_error = None
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_meta = {}

        while pending_cursor < len(pending) and len(future_to_meta) < workers:
            offset, index = pending[pending_cursor]
            future, meta = submit_one(executor, offset, index)
            future_to_meta[future] = meta
            pending_cursor += 1
            if SLEEP_BETWEEN_ITEMS > 0:
                time.sleep(SLEEP_BETWEEN_ITEMS)

        while future_to_meta:
            done_futures, _ = wait(future_to_meta.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                offset, index, item = future_to_meta.pop(future)
                try:
                    record = future.result()
                except SensitiveJudgeBlock as exc:
                    record = {
                        "index": index,
                        "item_hash": stable_hash(item),
                        "status": "skipped",
                        "skip_reason": "sensitive_words_detected",
                        "error": repr(exc),
                        "selected_source": None,
                        "selected_item": None,
                        "scores": [],
                        "usage": {"generation": {}, "judge": {}},
                    }
                    append_jsonl(args.checkpoint, record)
                    completed[index] = record
                    run_records.append(record)
                    done_count += 1
                    print(
                        f"[{offset}/{len(selected_indices)}] skipped index={index}: sensitive_words_detected",
                        flush=True,
                    )
                except Exception as exc:
                    append_jsonl(
                        args.error_log,
                        {
                            "index": index,
                            "item_hash": stable_hash(item),
                            "stage": "process_one",
                            "error": repr(exc),
                        },
                    )
                    first_error = exc
                    print(
                        f"[{offset}/{len(selected_indices)}] failed index={index}: {exc!r}. "
                        "Stop submitting new tasks; waiting for in-flight tasks to finish.",
                        flush=True,
                    )
                    continue

                else:
                    append_jsonl(args.checkpoint, record)
                    completed[index] = record
                    run_records.append(record)
                    done_count += 1

                if record.get("status") == "skipped":
                    pass
                elif PRINT_EVERY_ITEM:
                    original_score = record.get("scores", [{}])[0].get("final_score")
                    selected_score = None
                    for score_record in record.get("scores", []):
                        if score_record.get("source") == record.get("selected_source"):
                            selected_score = score_record.get("final_score")
                            break
                    print(
                        f"[{offset}/{len(selected_indices)}] done index={index}: "
                        f"selected={record.get('selected_source')} "
                        f"score={selected_score} original_score={original_score}",
                        flush=True,
                    )

                if done_count % SAVE_EVERY == 0:
                    usage = summarize_usage(completed[i] for i in selected_indices if i in completed)
                    print(f"processed {done_count}/{len(selected_indices)}, usage: {usage}", flush=True)

                if first_error is None and pending_cursor < len(pending):
                    next_offset, next_index = pending[pending_cursor]
                    next_future, next_meta = submit_one(executor, next_offset, next_index)
                    future_to_meta[next_future] = next_meta
                    pending_cursor += 1
                    if SLEEP_BETWEEN_ITEMS > 0:
                        time.sleep(SLEEP_BETWEEN_ITEMS)

        if first_error is not None:
            raise RuntimeError(
                "严格模式下检测到 API 失败，已停止提交新任务。"
                "已完成样本保留在 checkpoint，修复 API 后可重新运行断点续传。"
            ) from first_error

    output_data = []
    for index, item in enumerate(data):
        if index in selected_set:
            record = completed.get(index)
            if record and record.get("status") == "skipped":
                continue
            output_data.append(record["selected_item"] if record else item)
        elif args.limit is None and args.sample is None:
            output_data.append(item)

    if args.limit is not None or args.sample is not None:
        # 小规模试验默认只输出本次目标子集，避免误以为已经重排完整训练集。
        output_data = [
            completed[i]["selected_item"]
            for i in selected_indices
            if i in completed and completed[i].get("status") != "skipped"
        ]

    write_json(args.output, output_data)
    all_usage = summarize_usage(completed[i] for i in selected_indices if i in completed)
    skipped_count = sum(1 for i in selected_indices if i in completed and completed[i].get("status") == "skipped")
    print(f"saved: {args.output}")
    print(f"selected records: {len(output_data)}")
    print(f"skipped sensitive records: {skipped_count}")
    print(f"token usage observed from API responses: {all_usage}")


if __name__ == "__main__":
    main()
