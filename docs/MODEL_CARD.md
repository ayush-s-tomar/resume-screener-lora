> The same card is published at https://huggingface.co/Kus-hal/resume-screener-lora; it is kept here so it is versioned with the code and results.

# resume-screener-lora

A LoRA adapter for `Qwen/Qwen2.5-0.5B-Instruct` that reads a resume and a target role and returns a structured JSON verdict: an ATS-style score from 0 to 100, a verdict label (`strong_match`, `moderate_match` or `weak_match`) and lists of matched and missing skills. It is a research and portfolio artifact, not a hiring tool.

- Code, evaluation scripts and raw results: https://github.com/ayush-s-tomar/resume-screener-lora
- Live demo (it scores with a hosted 20B model, not with this adapter): https://resume-screener-finetune.onrender.com/
- Run the real adapter in a notebook: `demo_colab.ipynb` in the GitHub repository
- Quantized GGUF: https://huggingface.co/Kus-hal/resume-screener-gguf

## Model details

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Method | LoRA, rank 16, alpha 32, dropout 0.05 on `k_proj, o_proj, q_proj, v_proj` |
| Trainable parameters | about 2.16 million (0.44% of the weights) |
| Adapter size | 8.3 MB (base model 953 MB, GGUF `q4_k_m` 379 MB) |
| Language | English |
| Output | a JSON object with `ats_score` (0 to 100), a verdict label, `matched_skills` and `missing_skills` |
| Score bands in the training data | weak up to 48, moderate 50 to 74, strong 75 and above |

## Training data

800 synthetic training examples and 96 held-out synthetic examples, produced by a template-based generator with a written rubric. **The labels are the generator's, not human judgments.** The data is not published yet.

## Intended use

- Learning, demonstration and research on how to evaluate a small fine-tuned model honestly.
- Reproducing the evaluations in the GitHub repository.

## Out-of-scope use

- Any real hiring decision, automated rejection or ranking of real candidates.
- Processing real people's resumes without their consent.
- Non-English resumes, and roles or resume layouts that the evaluation did not cover.

## Evaluation

All numbers come from the files in `results/` in the GitHub repository (last checked against them on 2026-11-18).

**Held-out set (96 synthetic examples).** Intervals are 95% Wilson intervals.

| System | Verdict accuracy | Mean score error |
|---|---|---|
| Base model, zero-shot | 17.7% (17/96; 11 to 27%) | n/a |
| Base model with the rubric in the prompt | 18.8% (18/96; 12 to 28%) | n/a |
| gpt-oss-20b on Groq with the rubric (baseline script's prompt) | 55.2% (53/96; 45 to 65%) | 16.4 |
| **This adapter** | **88.5% (85/96; 81 to 93%)** | **5.4** |

This measures agreement with the generator's rubric on data from the same generator. It does not show that the adapter screens real resumes well.

**Hand-authored set (36 resumes in layouts the generator never produces).** The resumes and their labels were drafted with an AI assistant against a written rubric and have not been checked by humans.

- Verdict accuracy 44.4% (16/36), 2 severe errors (weak against strong), 1 output that was not valid JSON.
- The adapter is much more conservative here: it returned a weak match for 23 of 36 resumes (the rubric has 12) and a strong match only 3 times, all correct.
- Its scores still put the higher-labelled resume above the lower-labelled one in 80.4% of pairs, so the score carries signal even where the verdict thresholds do not transfer.
- The gap could come from the different writing style, from the rubric differing from the generator's rule, or both. Independent human labels have not been collected yet; an annotation form is in the repository.

**Name sensitivity.** On 15 held-out resumes with 8 different names, the mean absolute score change was 0.9 points (0.4 for a neutral placeholder) and 3 of 120 verdicts flipped, all on one resume 1.5 points from a verdict boundary. On the 8 resumes closest to a boundary, names flipped 3 of 64 verdicts and neutral prefixes (several of which contain digits) flipped 13 of 64. So the adapter is fragile close to a boundary, and in these tests names were not worse than other prefixes. This is a sensitivity check, not a fairness audit.

**Skill lists (held-out set, against the generator's lists).** Matched skills F1 1.00; missing skills F1 0.91 (precision 0.93, recall 0.89, exact set match 71.9%). 4.3% of the skills it lists as missing appear in the resume text (5.0% for the generator's own lists).

**Cost and latency.** One 16-thread Windows machine, CPU only: median 11.4 s per resume locally (p90 13.7 s, about 5.6 tokens per second, 2.18 GB resident, 16 s to load). The hosted demo took a median of 3.4 s (p90 5.5 s) per resume. The adapter's case is accuracy on its own task, offline use and no API key, not speed.

## Limitations

- Synthetic labels and small test sets (96 and 36 resumes), so every number carries wide uncertainty.
- Verdict thresholds learned on template resumes do not transfer well to differently written resumes (44.4% on the hand-authored set).
- Verdicts near a boundary (48 to 50 and 74 to 75) are unstable: small changes at the top of the prompt flipped them in the boundary test. Treat such scores as borderline.
- Output is not always valid JSON (1 of 36 on the hand-authored set).
- No human annotation yet, and no bias audit beyond the name checks above.

## Risks and regulation

Automated tools that screen candidates are regulated in some places (for example, New York City requires bias audits of automated employment decision tools, and the EU AI Act classifies recruitment systems as high risk). Check the current rules before using any such tool. This adapter has not been audited and must not be used for hiring.

## How to use

The exact prompt format is defined in `inference.py` in the GitHub repository. Use that script or the Colab notebook rather than writing your own prompt, because the adapter was trained on that format.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "Kus-hal/resume-screener-lora")
```

## Citation

```bibtex
@software{tomar2026resumescreener,
  author = {Tomar, Ayush Singh},
  title  = {resume-screener-lora: a LoRA fine-tuned resume screener with an honest evaluation},
  year   = {2026},
  url    = {https://github.com/ayush-s-tomar/resume-screener-lora}
}
```

## License

The adapter and the code are released under the MIT license. The base model has its own license; see its model card.
