# Model response testing

This repository fills `actual_output` fields from a question and its
`retrieval_context`, then evaluates the generated answers with DeepEval.

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

The workflow uses the first 10 entries from `dataset/responses.json`. The
response model generates answers with a 256-token cap and temperature 0.1.
DeepEval then uses `qwen3:4b-instruct` through Ollama at temperature 0.0 to
measure Faithfulness, Answer Relevancy, and Contextual Recall.

Expected outputs are joined from `dataset/golden_dataset.json` by question ID
for Contextual Recall. The committed datasets are not modified.

The job passes only when all 10 outputs and all 30 metric scores are present.
Metric scores below the 0.5 threshold are reported but do not fail CI. Logs and
partial results are uploaded even when generation or evaluation fails.

The artifact contains:

- `responses.json`: the 10 questions with generated outputs
- `summary.json`: response-generation timing and failed IDs
- `<model>.log`: readable per-question logs including generated answers
- `<model>.jsonl`: structured timing, token counts, answers, and Ollama metrics
- `deepeval-result.json`: raw DeepEval test and metric results
- `evaluation-summary.json`: per-question and aggregate metric scores
- `evaluation-report.md`: readable metric report and Confident AI link
- `deepeval-console.log`: complete DeepEval and verbose judge logs
- response and judge model pull logs and timings
- `ollama-server.log`: local model-server logs
- dependency, unit-test, and validation logs

If the repository secret `CONFIDENT_API_KEY` is configured, DeepEval also
publishes the test run to Confident AI. The local evaluation and GitHub artifact
do not require that secret.

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
  --max-tokens 256 \
  --temperature 0.1 \
  --output-dir results/qwen
```
