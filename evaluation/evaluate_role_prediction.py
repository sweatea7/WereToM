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


ROLE_MAPPING = {

    "werewolf": "狼人",

    "seer": "预言家",

    "witch": "女巫",

    "hunter": "猎人",

    "villager": "平民",

}


def get_cn_role(en_role):

    en_role_lower = str(en_role).lower()

    for en, cn in ROLE_MAPPING.items():

        if en in en_role_lower:

            return cn

    return "未知"


def get_camp(cn_role):

    if cn_role == "狼人":

        return "狼人阵营"

    if cn_role in ["预言家", "女巫", "猎人", "平民"]:

        return "好人阵营"

    return "未知阵营"


def extract_answer_content(text):

    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if not answer_match:

        return None

    return answer_match.group(1).strip()


def extract_valid_prediction_dict(ans_content):

    if not ans_content:

        return None


    try:

        strict_patterns = [

            r"^\s*[【\[]?\s*\d+\s*号?\s*[】\]]?\s*\n?\s*(\{.*\})\s*$",

            r"^\s*(?:投给|投票给|归票给|最终投给|决定投给)\s*[【\[]?\s*\d+\s*号?\s*[】\]]?\s*\n?\s*(\{.*\})\s*$",

        ]

        json_str = None

        for pattern in strict_patterns:

            match = re.fullmatch(pattern, ans_content, re.DOTALL)

            if match:

                json_str = match.group(1)

                break


        if json_str is None:

            return None


        preds = json.loads(json_str)

        if not isinstance(preds, dict) or not preds:

            return None


        valid_roles = {"狼人", "预言家", "女巫", "猎人", "平民", "村民", "好人"}

        valid_preds = {

            str(k).strip(): str(v).strip()

            for k, v in preds.items()

            if str(k).strip().isdigit() and str(v).strip() in valid_roles

        }

        if len(valid_preds) < 2:

            return None

        return valid_preds

    except json.JSONDecodeError:

        return None


def check_format_and_extract_json(text):

    answer_content = extract_answer_content(text)

    if answer_content is None:

        return False, {}


    has_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL) is not None

    if not has_think:

        return False, {}


    valid_preds = extract_valid_prediction_dict(answer_content)

    if valid_preds is None:

        return False, {}

    return True, valid_preds


def fallback_extract_roles_from_text(text, target_players):

    preds = {}

    for p in target_players:

        pattern = str(p) + r"(?:号玩家|号|[:：\-\s]{1,3}).{0,15}?([狼平预女猎][人民言巫][家]?)"

        match = re.search(pattern, text)

        if match:

            role_str = match.group(1)

            if "狼" in role_str:

                preds[p] = "狼人"

            elif "预" in role_str:

                preds[p] = "预言家"

            elif "女" in role_str:

                preds[p] = "女巫"

            elif "猎" in role_str:

                preds[p] = "猎人"

            elif "平" in role_str:

                preds[p] = "平民"

    return preds


def get_alive_players_from_input(input_text):

    match = re.search(r"【当前场上存活玩家】：\s*\[(.*?)\]", input_text)

    if match:

        players_str = match.group(1)

        return [p.strip() for p in players_str.split(",") if p.strip().isdigit()]

    return []


def resolve_model_dirs():

    if not os.path.exists(OUTPUT_BASE_DIR):

        return []


    model_dirs = sorted(

        [d for d in os.listdir(OUTPUT_BASE_DIR) if os.path.isdir(os.path.join(OUTPUT_BASE_DIR, d))]

    )

    return [d for d in model_dirs if d in SELECTED_MODEL_DIRS]


def evaluate_role_predict():

    if not os.path.exists(OUTPUT_BASE_DIR):

        print(f"❌ 找不到结果目录: {OUTPUT_BASE_DIR}")

        return


    model_dirs = resolve_model_dirs()

    summary_results = []


    for model_name in model_dirs:

        file_path = os.path.join(OUTPUT_BASE_DIR, model_name, "results_vote_role_predict.json")

        if not os.path.exists(file_path):

            continue


        with open(file_path, "r", encoding="utf-8") as f:

            data = json.load(f)


        total_samples = len(data)

        format_pass_count = 0


        global_metrics = {

            "exact": {"hits_valid": 0, "hits_all": 0, "dv": 0, "da": 0},

            "camp": {"hits_valid": 0, "hits_all": 0, "dv": 0, "da": 0},

        }

        role_stats = {k: {"tp": 0, "pp": 0, "ap": 0} for k in ["wolf", "seer", "witch", "hunter"]}

        role_key_map = {"狼人": "wolf", "预言家": "seer", "女巫": "witch", "猎人": "hunter"}


        for item in data:

            model_output = item.get("model_output", "")

            roles_map = item.get("roles_map", {})

            ai_seat = str(item.get("player_id"))

            input_text = item.get("input", "")


            alive_players = get_alive_players_from_input(input_text)

            if not alive_players:

                alive_players = list(roles_map.keys())

            target_players = [p for p in alive_players if str(p) != ai_seat]


            is_format_valid, preds_dict = check_format_and_extract_json(model_output)

            if is_format_valid:

                format_pass_count += 1

            else:

                preds_dict = fallback_extract_roles_from_text(model_output, target_players)


            for target_seat in target_players:

                t_str = str(target_seat)

                true_role_en = roles_map.get(t_str, roles_map.get(int(target_seat), "Unknown"))

                true_role_cn = get_cn_role(true_role_en)

                true_camp = get_camp(true_role_cn)


                pred_role = str(preds_dict.get(t_str, preds_dict.get(int(target_seat), ""))).strip()

                pred_camp = get_camp(pred_role) if pred_role else "无"


                global_metrics["exact"]["da"] += 1

                global_metrics["camp"]["da"] += 1

                if is_format_valid:

                    global_metrics["exact"]["dv"] += 1

                    global_metrics["camp"]["dv"] += 1


                if pred_role == true_role_cn:

                    global_metrics["exact"]["hits_all"] += 1

                    if is_format_valid:

                        global_metrics["exact"]["hits_valid"] += 1


                if pred_camp == true_camp:

                    global_metrics["camp"]["hits_all"] += 1

                    if is_format_valid:

                        global_metrics["camp"]["hits_valid"] += 1


                tk, pk = role_key_map.get(true_role_cn), role_key_map.get(pred_role)

                if tk:

                    role_stats[tk]["ap"] += 1

                if pk:

                    role_stats[pk]["pp"] += 1

                if tk and tk == pk:

                    role_stats[tk]["tp"] += 1


        res = {"model": model_name}

        res["fmt_rate"] = (format_pass_count / total_samples) * 100 if total_samples > 0 else 0


        for k in ["exact", "camp"]:

            h_v, h_a = global_metrics[k]["hits_valid"], global_metrics[k]["hits_all"]

            dv, da = global_metrics[k]["dv"], global_metrics[k]["da"]

            res[f"{k}_valid"] = (h_v / dv * 100) if dv > 0 else 0

            res[f"{k}_all"] = (h_a / da * 100) if da > 0 else 0


        for k in ["wolf", "seer", "witch", "hunter"]:

            tp, pp, ap = role_stats[k]["tp"], role_stats[k]["pp"], role_stats[k]["ap"]

            p = (tp / pp * 100) if pp > 0 else 0.0

            r = (tp / ap * 100) if ap > 0 else 0.0

            res[f"{k}_p"] = p

            res[f"{k}_r"] = r

            res[f"{k}_f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


        summary_results.append(res)


    if not summary_results:

        print("❌ 没有找到可评测结果。")

        return


    max_fmt = max(r["fmt_rate"] for r in summary_results)

    max_e_all = max(r["exact_all"] for r in summary_results)

    max_c_all = max(r["camp_all"] for r in summary_results)

    max_f1 = {k: max(r[f"{k}_f1"] for r in summary_results) for k in ["wolf", "seer", "witch", "hunter"]}


    widths = [24, 12, 18, 18, 25, 25, 25, 25]

    headers = ["Model Name", "Fmt Rate", "Exact Acc", "Camp Acc", "Wolf P/R (F1)", "Seer P/R (F1)", "Witch P/R (F1)", "Hunter P/R (F1)"]


    total_len = sum(widths) + len(widths) * 3 + 1

    print("\n" + "=" * total_len)

    print("📊 WereToM evaluation: role prediction metrics P/R/F1")

    print("=" * total_len)


    header_str = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"

    print(header_str)

    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")


    for r in summary_results:

        model_str = r["model"].ljust(widths[0])


        fmt_val = f"{r['fmt_rate']:.1f}%"

        fmt_s = f"*{fmt_val}*" if (r["fmt_rate"] == max_fmt and max_fmt > 0) else fmt_val

        fmt_s = fmt_s.ljust(widths[1])


        def acc_s(key, max_val, width):

            base_s = f"{r[f'{key}_valid']:.1f}%/{r[f'{key}_all']:.1f}%"

            final_s = f"*{base_s}*" if (r[f"{key}_all"] == max_val and max_val > 0) else base_s

            return final_s.ljust(width)


        def prf1_s(key, width):

            base_s = f"{r[f'{key}_p']:.1f}%/{r[f'{key}_r']:.1f}% ({r[f'{key}_f1']:.1f})"

            final_s = f"*{base_s}*" if (r[f"{key}_f1"] == max_f1[key] and max_f1[key] > 0) else base_s

            return final_s.ljust(width)


        row_str = (

            f"| {model_str} | {fmt_s} | {acc_s('exact', max_e_all, widths[2])} | "

            f"{acc_s('camp', max_c_all, widths[3])} | {prf1_s('wolf', widths[4])} | "

            f"{prf1_s('seer', widths[5])} | {prf1_s('witch', widths[6])} | {prf1_s('hunter', widths[7])} |"

        )

        print(row_str)


    print("=" * total_len + "\n")


if __name__ == "__main__":

    evaluate_role_predict()
