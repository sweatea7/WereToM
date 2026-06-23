# WereToM Anonymous Release

This repository contains anonymized code and data samples for the WereToM benchmark.

## Contents

- `dataset_construction/rerank_vote_predict_dataset.py`: generates multiple candidate answers for each first-person Werewolf voting sample and reranks them with local rules and an optional LLM judge.
- `dataset_construction/repair_low_score_dataset.py`: repairs low-score records after reranking. Records below 60 are discarded by default; records in the 60-80 range are regenerated and accepted only when the repaired answer improves the score and passes the configured threshold.
- `evaluation/`: evaluation scripts for decision accuracy, role prediction, and explicit-ToM reasoning quality.
- `data/weretom_test_924.json`: the 924-sample test split used for benchmark evaluation.

The full training split is not included in this anonymous package at this stage. The released file is intended to document the benchmark format and support inspection of the evaluation split.

## Dataset Format

Each JSON record follows an instruction-tuning format:

- `instruction`: task instruction for first-person Werewolf decision making.
- `input`: observable game context for the acting player.
- `output`: target response containing explicit ToM reasoning in `<think>...</think>` and final vote plus role prediction in `<answer>...</answer>`.

## Running the Construction Scripts

Install dependencies:

```bash
pip install openai
```

Run a small reranking check:

```bash
python dataset_construction/rerank_vote_predict_dataset.py --limit 5 --dry-run
```

Run reranking with API-backed generation and judging:

```bash
GENERATOR_API_KEY=your_generator_key \
JUDGE_API_KEY=your_judge_key \
JUDGE_BASE_URL=your_judge_base_url \
JUDGE_MODEL=your_judge_model \
python dataset_construction/rerank_vote_predict_dataset.py --limit 20
```

Run low-score repair:

```bash
GENERATOR_API_KEY=your_generator_key \
JUDGE_API_KEY=your_judge_key \
JUDGE_BASE_URL=your_judge_base_url \
JUDGE_MODEL=your_judge_model \
python dataset_construction/repair_low_score_dataset.py --limit 20
```

Do not commit API keys, local absolute paths, checkpoints, or error logs.

## Evaluation Scripts

The evaluation scripts expect model outputs under `WERETOM_RESULT_DIR` with one subdirectory per model. Each model directory should contain `results_vote_role_predict.json`.

```text
experiment_results/
├── Model-A/
│   └── results_vote_role_predict.json
└── Model-B/
    └── results_vote_role_predict.json
```

Available scripts:

- `evaluation/evaluate_good_vote.py`: good-camp vote accuracy.
- `evaluation/evaluate_wolf_vote_consistency.py`: wolf-camp vote consistency against human votes. This script additionally requires `WERETOM_TRUE_VOTE_DATASET`.
- `evaluation/evaluate_role_prediction.py`: role-prediction precision, recall, and F1.
- `evaluation/evaluate_think_components.py`: LLM-judge scores for ToM1, ToM2, Strategy, and Think final.
- `evaluation/evaluate_tom1_input_consistency.py`: hybrid local-rule and LLM-judge score for ToM1-input consistency.
- `evaluation/evaluate_strategy_answer_consistency.py`: hybrid local-rule and LLM-judge score for Strategy-answer consistency.

Example:

```bash
WERETOM_RESULT_DIR=experiment_results python evaluation/evaluate_good_vote.py
```

For LLM-judge evaluation:

```bash
WERETOM_RESULT_DIR=experiment_results \
DEEPSEEK_API_KEY=your_api_key \
DEEPSEEK_BASE_URL=https://api.deepseek.com \
DEEPSEEK_MODEL=deepseek-v4-flash \
python evaluation/evaluate_think_components.py
```

For wolf-vote consistency:

```bash
WERETOM_RESULT_DIR=experiment_results \
WERETOM_TRUE_VOTE_DATASET=data/weretom_test_924_with_true_vote.json \
python evaluation/evaluate_wolf_vote_consistency.py
```
