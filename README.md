# Qwen response testing with a Gemma QAT judge

The evaluation workflow uses one fixed model combination:

- Response model: `qwen3:4b-instruct` (`Q4_K_M`)
- Judge model: `gemma3:4b-it-qat`
- Metrics: Faithfulness, Answer Relevancy, and Contextual Recall

## Workflow

The expensive workflow runs on a pull request only while it has the `eval`
label. Every run processes all 50 questions as five parallel shards of 10.
Each shard generates fresh responses and then starts DeepEval immediately.

Successful scores are retained. If a judge call returns malformed JSON or
times out, the shard retries only the missing `(question, metric)` pairs. It
does not regenerate responses or rerun metrics that already returned scores.

A prerequisite job restores the quantized Ollama model cache. On the first
run, it pulls Qwen and Gemma once and saves the populated cache before the
matrix starts. Every shard then restores the completed cache and only verifies
the model manifests. Python dependencies use the `setup-python` pip cache.

## Quality gate

All 50 responses must be present. Each metric must independently satisfy:

- At least 90% score coverage
- Average score of at least 0.5
- At least 90% of evaluated scores at or above 0.5

Missing judge scores remain visible in the report as warnings. They fail CI
only when one of the coverage or quality conditions is not met.

## Reason trials

`configs/evaluation.json` controls `include_reason` and the judge output limit.
The controlled comparison uses:

| Trial | Reasons | Judge output tokens |
| --- | ---: | ---: |
| No reasons | false | 1024 |
| Reasons | true | 2048 |

Both trials keep the dataset, response prompt, model tags, temperatures,
context windows, metrics, thresholds, and sharding identical.

## Sampling parameters

| Setting | Response | Judge |
| --- | ---: | ---: |
| Temperature | 0.1 | 0.0 |
| Output tokens | 256 | trial-dependent |
| Context window | 16384 | 16384 |
| Seed | 42 | 42 |

## Artifacts

Every shard uploads generated responses, generation timings, raw DeepEval
results, retry results, metric summaries, and model logs. The final artifact
contains the combined `responses.json`, `evaluation-summary.json`, and
`evaluation-report.md`.

## Local validation

```bash
python -m unittest discover -s tests -v
python scripts/build_shard_matrix.py \
  --dataset dataset/responses.json \
  --question-count 50 \
  --shard-size 10
```
