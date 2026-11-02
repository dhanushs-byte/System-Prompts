# Week 4 Day 3 — System Prompts, Zero-Shot and Few-Shot

## Included
- `data/tickets.csv`: 10 sample support tickets with ground-truth category and priority.
- `prompts/v1.txt`, `v2.txt`, `v3.txt`: prompt versions.
- `prompts/few_shot_examples.json`: curated 2, 3 and 5-shot examples.
- `results/`: illustrative sample output CSVs so the submission has visible example results.
- `run_experiment.py`: OpenAI API runner from the original project (requires API credits).
- `dynamic_examples.py`: local Sentence Transformers bonus implementation.

## Important about the included output
Files ending in `_illustrative.csv` are demonstration/sample results, not measured model runs.
They show the expected table layout and example calculations. Do not report them as actual
API/model measurements. Run the scripts with a working model/API to generate genuine results.

## Run the local embedding bonus
```powershell
pip install sentence-transformers pandas numpy
python dynamic_examples.py
```
The first run downloads `all-MiniLM-L6-v2`. Subsequent embedding generation runs locally.
The dynamic selection script writes `results/dynamic_example_selection.csv`.

## Prompt ablation
- v1: complete prompt with role, task, tone, constraints and JSON output format.
- v2: removes only the output-format instruction.
- v3: removes only the constraints section.

## Four lines: when few-shot is better than zero-shot
1. Few-shot helps when category boundaries are subtle or ambiguous.
2. Examples demonstrate the exact labels and response behavior expected.
3. Representative examples can improve consistency on domain-specific tickets.
4. Zero-shot is useful when token cost, latency, or labelled examples are limited.

## Example order / wrong example
Reverse the order of examples and rerun to test order sensitivity. Then deliberately add
one incorrect demonstration (e.g., label a duplicate-payment ticket as Shipping/Low) and
compare predictions. A wrong example can bias the classifier toward an incorrect label.

## Git
```bash
git add .
git commit -m "Add Week 4 Day 3 prompt engineering tasks and sample outputs"
git push
```
