import json

import os

import re


OUTPUT_BASE_DIR = os.getenv("WERETOM_RESULT_DIR", "experiment_results")

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


def resolve_model_dirs():

    if not os.path.exists(OUTPUT_BASE_DIR):

        return []


    model_dirs = sorted(

        [d for d in os.listdir(OUTPUT_BASE_DIR) if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]

    )

    return [d for d in model_dirs if d in SELECTED_MODEL_DIRS]


def evaluate_vote():

    if not os.path.exists(OUTPUT_BASE_DIR):

        print(f"❌ 找不到结果目录: {OUTPUT_BASE_DIR}")

        return


    model_dirs = resolve_model_dirs()

    summary_results = []

    human_acc_record = 0.0


    for model_name in model_dirs:

        file_path = os.path.join(OUTPUT_BASE_DIR, model_name, "results_vote_role_predict.json")

        if not os.path.exists(file_path):

            continue


        with open(file_path, "r", encoding="utf-8") as f:

            data = json.load(f)


        total_samples = len(data)

        format_pass_count = 0


        good_voters_total = 0

        good_voters_format_pass = 0

        good_voters_format_non_abstain = 0


        human_correct_votes = 0

        model_correct_votes_all = 0

        model_correct_votes_valid = 0

        model_correct_votes_valid_non_abstain = 0

        abstain_count = 0


        for item in data:

            human_output = item.get("output", "")

            model_output = item.get("model_output", "")

            roles_map = item.get("roles_map", {})

            ai_role = item.get("player_role", "Unknown")


            is_format_valid = check_format(model_output)

            if is_format_valid:

                format_pass_count += 1

            is_model_abstain = is_abstain(model_output)

            if is_model_abstain:

                abstain_count += 1


            if is_wolf(ai_role):

                continue


            good_voters_total += 1

            if is_format_valid:

                good_voters_format_pass += 1

                if not is_model_abstain:

                    good_voters_format_non_abstain += 1


            human_target = extract_vote_target(human_output)

            if human_target and human_target in roles_map and is_wolf(roles_map[human_target]):

                human_correct_votes += 1


            model_target = extract_vote_target(model_output)

            if model_target and model_target in roles_map and is_wolf(roles_map[model_target]):

                model_correct_votes_all += 1

                if is_format_valid:

                    model_correct_votes_valid += 1

                    if not is_model_abstain:

                        model_correct_votes_valid_non_abstain += 1


        fmt_rate = (format_pass_count / total_samples) * 100 if total_samples > 0 else 0

        human_acc = (human_correct_votes / good_voters_total) * 100 if good_voters_total > 0 else 0

        human_acc_record = human_acc


        model_acc_valid = (model_correct_votes_valid / good_voters_format_pass) * 100 if good_voters_format_pass > 0 else 0

        model_acc_all = (model_correct_votes_all / good_voters_total) * 100 if good_voters_total > 0 else 0

        model_acc_valid_non_abstain = (

            (model_correct_votes_valid_non_abstain / good_voters_format_non_abstain) * 100

            if good_voters_format_non_abstain > 0

            else 0

        )


        summary_results.append(

            {

                "model": model_name,

                "fmt_rate": fmt_rate,

                "acc_valid": model_acc_valid,

                "acc_all": model_acc_all,

                "acc_valid_non_abstain": model_acc_valid_non_abstain,

                "abstain_count": abstain_count,

                "good_total": good_voters_total,

            }

        )


    if not summary_results:

        print("❌ 没有找到可评测结果。")

        return


    max_fmt = max(r["fmt_rate"] for r in summary_results)

    max_acc_valid = max(r["acc_valid"] for r in summary_results)

    max_acc_all = max(r["acc_all"] for r in summary_results)

    max_acc_valid_non_abstain = max(r["acc_valid_non_abstain"] for r in summary_results)


    widths = [24, 18, 19, 17, 25, 14]

    headers = [

        "Model Name",

        "Format Pass Rate",

        "Model Acc (Valid)",

        "Model Acc (All)",

        "Acc (Valid Non-Abstain)",

        "Abstain Count",

    ]


    total_len = sum(widths) + len(widths) * 3 + 1

    print("\n" + "=" * total_len)

    print(f"📊 WereToM evaluation: good-camp vote accuracy - {summary_results[0]['good_total']} good-camp samples")

    print("=" * total_len)

    print(f"🧍 人类基线表现 (Human Baseline): 准确率 {human_acc_record:.1f}%")

    print("-" * total_len)


    header_str = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    print(header_str)

    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")


    def acc_s(val, max_val, width):

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

            f"{acc_s(r['acc_valid'], max_acc_valid, widths[2])} | "

            f"{acc_s(r['acc_all'], max_acc_all, widths[3])} | "

            f"{acc_s(r['acc_valid_non_abstain'], max_acc_valid_non_abstain, widths[4])} | "

            f"{str(r['abstain_count']).ljust(widths[5])} |"

        )

        print(row_str)


    print("=" * total_len + "\n")


if __name__ == "__main__":

    evaluate_vote()
