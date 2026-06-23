import json

import os

import re


OUTPUT_BASE_DIR = os.getenv("WERETOM_RESULT_DIR", "experiment_results")

TRUE_VOTE_DATASET = os.getenv("WERETOM_TRUE_VOTE_DATASET", "data/weretom_test_924_with_true_vote.json")

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


def is_wolf(role_en):

    return "werewolf" in str(role_en).lower() or "狼" in str(role_en)


def extract_vote_target(text):

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if not answer_match:

        return None

    clean_ans = answer_match.group(1).strip()

    vote_part = re.split(r"\{", clean_ans, maxsplit=1)[0].strip()

    vote_match = re.search(r"(\d+)", vote_part)

    if vote_match:

        return str(vote_match.group(1))

    return None


def extract_valid_prediction_dict(clean_ans):

    if not clean_ans:

        return None


    strict_patterns = [

        r"^\s*[【\[]?\s*\d+\s*号?\s*[】\]]?\s*\n?\s*(\{.*\})\s*$",

        r"^\s*(?:投给|投票给|归票给|最终投给|决定投给)\s*[【\[]?\s*\d+\s*号?\s*[】\]]?\s*\n?\s*(\{.*\})\s*$",

    ]


    json_str = None

    for pattern in strict_patterns:

        match = re.fullmatch(pattern, clean_ans, re.DOTALL)

        if match:

            json_str = match.group(1)

            break


    if json_str is None:

        return None


    try:

        preds = json.loads(json_str)

    except json.JSONDecodeError:

        return None


    if not isinstance(preds, dict) or not preds:

        return None


    valid_roles = {"狼人", "预言家", "女巫", "猎人", "平民", "村民", "好人"}

    valid_pairs = 0

    for seat, role in preds.items():

        if not str(seat).strip().isdigit():

            continue

        if str(role).strip() in valid_roles:

            valid_pairs += 1

    return preds if valid_pairs >= 2 else None


def check_format(text):

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if not answer_match:

        return False


    has_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL) is not None

    if not has_think:

        return False


    clean_ans = answer_match.group(1).strip()

    return extract_vote_target(text) is not None and extract_valid_prediction_dict(clean_ans) is not None


def is_abstain(text):

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if not answer_match:

        return False

    clean_ans = answer_match.group(1).strip()

    return any(marker in clean_ans for marker in ["弃票", "过票", "不投", "无法投票"])


def normalize_text(value):

    return str(value or "").strip()


def normalize_context_input(input_text):

    text = normalize_text(input_text)

    if "【行动要求】" in text:

        text = text.split("【行动要求】", 1)[0].strip()

    return text


def normalize_role_map(role_map):

    if not isinstance(role_map, dict):

        return ""

    return json.dumps(role_map, ensure_ascii=False, sort_keys=True)


def build_true_vote_lookup():

    if not os.path.exists(TRUE_VOTE_DATASET):

        raise FileNotFoundError(f"找不到真实投票数据集: {TRUE_VOTE_DATASET}")


    with open(TRUE_VOTE_DATASET, "r", encoding="utf-8") as f:

        data = json.load(f)


    lookup = {}

    for item in data:

        true_vote_info = item.get("_true_vote") or {}

        key = (

            normalize_context_input(item.get("input")),

            normalize_role_map(item.get("roles_map")),

            normalize_text(true_vote_info.get("self_seat")),

            normalize_text(true_vote_info.get("self_role")),

        )

        lookup[key] = true_vote_info

    return lookup


def resolve_model_dirs():

    if not os.path.exists(OUTPUT_BASE_DIR):

        return []


    model_dirs = sorted(

        [d for d in os.listdir(OUTPUT_BASE_DIR) if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]

    )

    return [d for d in model_dirs if d in SELECTED_MODEL_DIRS]


def evaluate_wolf_vote_consistency():

    if not os.path.exists(OUTPUT_BASE_DIR):

        print(f"❌ 找不到结果目录: {OUTPUT_BASE_DIR}")

        return


    try:

        true_vote_lookup = build_true_vote_lookup()

    except FileNotFoundError as e:

        print(f"❌ {e}")

        return


    model_dirs = resolve_model_dirs()

    summary_results = []

    dataset_output_consistency_record = 0.0


    for model_name in model_dirs:

        file_path = os.path.join(OUTPUT_BASE_DIR, model_name, "results_vote_role_predict.json")

        if not os.path.exists(file_path):

            continue


        with open(file_path, "r", encoding="utf-8") as f:

            data = json.load(f)


        total_samples = len(data)

        format_pass_count = 0


        wolf_voters_total = 0

        wolf_voters_format_pass = 0

        wolf_voters_format_non_abstain = 0


        dataset_output_consistent = 0

        model_consistent_all = 0

        model_consistent_valid = 0

        model_consistent_valid_non_abstain = 0

        abstain_count = 0


        for item in data:

            model_output = item.get("model_output", "")

            ai_role = item.get("player_role", "Unknown")

            lookup_key = (

                normalize_context_input(item.get("input")),

                normalize_role_map(item.get("roles_map")),

                normalize_text(item.get("player_id")),

                normalize_text(item.get("player_role")),

            )

            true_vote_info = true_vote_lookup.get(lookup_key, {})


            is_format_valid = check_format(model_output)

            if is_format_valid:

                format_pass_count += 1

            is_model_abstain = is_abstain(model_output)

            if is_model_abstain:

                abstain_count += 1


            if not is_wolf(ai_role):

                continue


            wolf_voters_total += 1

            if is_format_valid:

                wolf_voters_format_pass += 1

                if not is_model_abstain:

                    wolf_voters_format_non_abstain += 1


            human_target = normalize_text(true_vote_info.get("true_vote_target"))

            if human_target is not None:

                if human_target == "":

                    human_target = None

            dataset_output_target = extract_vote_target(item.get("output", ""))

            if human_target is not None and dataset_output_target is not None and dataset_output_target == human_target:

                dataset_output_consistent += 1


            model_target = extract_vote_target(model_output)

            if model_target is not None and human_target is not None and model_target == human_target:

                model_consistent_all += 1

                if is_format_valid:

                    model_consistent_valid += 1

                    if not is_model_abstain:

                        model_consistent_valid_non_abstain += 1


        fmt_rate = (format_pass_count / total_samples) * 100 if total_samples > 0 else 0

        dataset_output_consistency = (

            (dataset_output_consistent / wolf_voters_total) * 100 if wolf_voters_total > 0 else 0

        )

        dataset_output_consistency_record = dataset_output_consistency


        model_consistency_valid = (

            (model_consistent_valid / wolf_voters_format_pass) * 100 if wolf_voters_format_pass > 0 else 0

        )

        model_consistency_all = (

            (model_consistent_all / wolf_voters_total) * 100 if wolf_voters_total > 0 else 0

        )

        model_consistency_valid_non_abstain = (

            (model_consistent_valid_non_abstain / wolf_voters_format_non_abstain) * 100

            if wolf_voters_format_non_abstain > 0

            else 0

        )


        summary_results.append(

            {

                "model": model_name,

                "fmt_rate": fmt_rate,

                "cons_valid": model_consistency_valid,

                "cons_all": model_consistency_all,

                "cons_valid_non_abstain": model_consistency_valid_non_abstain,

                "abstain_count": abstain_count,

                "wolf_total": wolf_voters_total,

            }

        )


    if not summary_results:

        print("❌ 没有找到可评测结果。")

        return


    max_fmt = max(r["fmt_rate"] for r in summary_results)

    max_cons_valid = max(r["cons_valid"] for r in summary_results)

    max_cons_all = max(r["cons_all"] for r in summary_results)

    max_cons_valid_non_abstain = max(r["cons_valid_non_abstain"] for r in summary_results)


    widths = [24, 18, 22, 20, 28, 14]

    headers = [

        "Model Name",

        "Format Pass Rate",

        "Vote Cons (Valid)",

        "Vote Cons (All)",

        "Cons (Valid Non-Abstain)",

        "Abstain Count",

    ]


    total_len = sum(widths) + len(widths) * 3 + 1

    print("\n" + "=" * total_len)

    print(f"📊 WereToM evaluation: wolf vote consistency - {summary_results[0]['wolf_total']} wolf-camp samples")

    print("=" * total_len)

    print(

        f"🧍 数据集输出基线 (Dataset Output vs True Human Vote): 一致性 {dataset_output_consistency_record:.1f}%"

    )

    print("-" * total_len)


    header_str = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    print(header_str)

    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")


    def rate_s(val, max_val, width):

        base_s = f"{val:.1f}%"

        final_s = f"*{base_s}*" if (val == max_val and max_val > 0) else base_s

        return final_s.ljust(width)


    for r in summary_results:

        model_str = r["model"].ljust(widths[0])


        fmt_val = f"{r['fmt_rate']:.1f}%"

        fmt_s = f"*{fmt_val}*" if (r["fmt_rate"] == max_fmt and max_fmt > 0) else fmt_val

        fmt_s = fmt_s.ljust(widths[1])


        row_str = (

            f"| {model_str} | {fmt_s} | "

            f"{rate_s(r['cons_valid'], max_cons_valid, widths[2])} | "

            f"{rate_s(r['cons_all'], max_cons_all, widths[3])} | "

            f"{rate_s(r['cons_valid_non_abstain'], max_cons_valid_non_abstain, widths[4])} | "

            f"{str(r['abstain_count']).ljust(widths[5])} |"

        )

        print(row_str)


    print("=" * total_len + "\n")


if __name__ == "__main__":

    evaluate_wolf_vote_consistency()
