# Model response testing

This repository fills `actual_output` fields from a question and its
`retrieval_context`.

## Model branches

Pull requests trigger response generation only when their base branch is one
of the model branches below. Provider-specific model identifiers are defined in
`configs/models.json`.

| Base branch | Response model | Provider | Model identifier |
| --- | --- | --- | --- |
| `qwen` | Qwen3 4B Instruct | Ollama | `qwen3:4b-instruct` |
| `gemma` | Gemma 4 E4B IT | Hugging Face Transformers | `google/gemma-4-E4B-it` |
| `phi` | Phi-4 Mini Instruct | Ollama | `phi4-mini` |

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
- `<model>.jsonl`: structured timing, token counts, answers, and provider metrics
- provider install and model download logs
- `ollama-server.log`: local model-server logs for Ollama-backed models
- `validation.log`: final output validation result


## Local checks

Run its tests with:

```bash
python -m unittest discover -s tests -v
```

When Ollama is already running locally, a response run can be started with:

```bash
python runners/run_responses.py \
  --model qwen3:4b-instruct \
  --model-name qwen \
  --provider ollama \
  --limit 10 \
  --max-tokens 512 \
  --temperature 0.1 \
  --output-dir results/qwen
```

For Gemma 4 through Hugging Face Transformers:

```bash
python -m pip install -U transformers torch accelerate sentencepiece
python runners/run_responses.py \
  --model google/gemma-4-E4B-it \
  --model-name gemma \
  --provider huggingface \
  --limit 10 \
  --max-tokens 512 \
  --temperature 0.1 \
  --output-dir results/gemma
```
