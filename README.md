# Qwen response testing with a Gemma judge

This branch runs one fixed evaluation combination:

- Response model: `qwen3:4b-instruct` (`Q4_K_M`)
- Judge model: `gemma3:4b-it-qat` (quantization-aware trained)
- Metrics: Faithfulness, Answer Relevancy, and Contextual Recall

## Question count and sharding

The default question count and shard size are defined in
`configs/evaluation.json`. Pull request runs use that default unless the
repository variable `EVAL_QUESTION_COUNT` is set. A manual workflow run can
override it with the optional `question_count` input.

Questions are divided into shards of 10. Each matrix job generates its shard
with Qwen and immediately starts DeepEval with Gemma. Other shards continue in
parallel. After every shard finishes, the aggregate job combines responses,
per-question scores, weighted metric averages, timings, and errors into one
report.

Examples:

- `10` questions creates one shard.
- `25` questions creates shards of 10, 10, and 5.
- `50` questions creates five parallel shards of 10.

## Sampling parameters

| Setting | Response model | Judge model |
| --- | ---: | ---: |
| Temperature | 0.1 | 0.0 |
| Maximum output tokens | 256 | 2048 |
| Context window | 16384 | 16384 |
| Seed | 42 | 42 |

The larger judge output limit prevents DeepEval JSON and metric reasons from
being cut off. Metric threshold is `0.5`; a low score is reported, while a
missing or invalid score fails the shard and final aggregation.

## Artifacts

Every shard uploads generated responses, generation timing, raw DeepEval
results, metric summaries, model logs, and console logs. The final artifact
contains:

- `responses.json`: all generated responses in dataset order
- `evaluation-summary.json`: aggregate and per-question scores
- `evaluation-report.md`: the readable final report

## Local validation

```bash
python -m unittest discover -s tests -v
python scripts/build_shard_matrix.py \
  --dataset dataset/responses.json \
  --question-count 50 \
  --shard-size 10
```
