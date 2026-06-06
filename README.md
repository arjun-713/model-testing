# Model response testing

This repository fills `actual_output` fields from a question and its
`retrieval_context`. Judge-model evaluation with DeepEval will be added after
the response-model runs are stable.

## Model branches

Pull requests trigger response generation only when their base branch is one
of the model branches below. The exact Ollama tags are defined in
`configs/models.json`.

| Base branch | Response model | Ollama tag |
| --- | --- | --- |
| `qwen` | Qwen3 4B Instruct | `qwen3:4b-instruct` |
| `gemma` | Gemma 3n E4B | `gemma3n:e4b` |
| `phi` | Phi-4 Mini Instruct | `phi4-mini` |

To test a model, create a small change on a separate branch and open a pull
request whose base branch is the model branch. A pull request targeting `main`
does not run the model workflow.

## CI behavior

The workflow splits all 50 entries from `dataset/responses.json` into five
parallel shards of 10 questions. Each answer is capped at 256 tokens and uses a
temperature of 0.1. It does not modify the committed source dataset.

Each shard runs on a separate GitHub-hosted runner and pulls its own copy of the
selected model. An aggregation job rebuilds the 50 responses in source order.
The workflow passes only when every `actual_output` is non-empty. Logs and
partial results are uploaded even when generation or validation fails.

Each shard artifact contains:

- `responses.json`: that shard's 10 questions with generated outputs
- `summary.json`: total and average response time, fill count, and failed IDs
- `<model>.log`: readable logs including generated answers
- `<model>.jsonl`: timing, token counts, and Ollama metrics
- `model-pull.log` and `model-pull-timing.json`: model download details
- `ollama-server.log`: local model-server logs
- `validation.log`: final output validation result

The combined root artifact contains:

- `responses.json`: all 50 responses in the original dataset order
- `summary.json`: parallel and summed timings, model pull time, and speedup
- `report.md`: a readable timing report also shown in the workflow summary
- `combined.jsonl`: structured events from all five shards
- `shards/`: complete copies of every shard's files and logs

## Local checks

The runner uses only the Python standard library. Run its tests with:

```bash
python -m unittest discover -s tests -v
```

When Ollama is already running locally, a response run can be started with:

```bash
python runners/run_responses.py \
  --model qwen3:4b-instruct \
  --model-name qwen \
  --offset 0 \
  --limit 10 \
  --max-tokens 256 \
  --temperature 0.1 \
  --output-dir results/qwen
```
