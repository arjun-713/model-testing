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

The workflow uses the first 10 entries from `dataset/responses.json`, caps each
generation at 512 tokens, and writes a new response file under
`results/<model-branch>/responses.json`. It does not modify the committed
50-question source dataset.

The job passes only when all 10 selected entries contain a non-empty
`actual_output`. Logs and partial results are uploaded even when generation or
validation fails.

The artifact contains:

- `responses.json`: the 10 questions with generated outputs
- `summary.json`: total and average response time, fill count, and failed IDs
- `<model>.log`: readable per-question logs including generated answers
- `<model>.jsonl`: structured timing, token counts, answers, and Ollama metrics
- `model-pull.log` and `model-pull-timing.json`: model download details
- `ollama-server.log`: local model-server logs
- `validation.log`: final output validation result

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
  --limit 10 \
  --max-tokens 512 \
  --output-dir results/qwen
```
