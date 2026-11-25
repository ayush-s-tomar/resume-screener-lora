# Privacy

> Written by `docs/build_docs.py` on 2026-11-25 from `render-deploy/app.py`. It describes what this project does with data. It is not legal advice.

## Summary

- **The hosted demo sends the text you submit to Groq's API (model `openai/gpt-oss-20b`) to be scored.** Groq processes it under its own terms. Please do not submit real personal data.
- **The local adapter runs on your machine.** The resume text does not leave it. The model files are downloaded from Hugging Face on first use.
- The demo page has no accounts, no cookies and no analytics or other external scripts.
- This repository contains only synthetic or fictional resumes.

## Hosted demo

### What the server keeps

| Item | Content | Lifetime |
|---|---|---|
| Result cache | The verdict, score, skills and rationale, keyed by a SHA-256 hash of the role and text. The text itself is not stored. The rationale is written by the model and may paraphrase the resume. | In memory, up to 256 entries, cleared whenever the server restarts |
| Rate counters | Timestamps of recent requests per IP address. Limit: 8 new screenings per visitor per 10 minutes and 150 per day overall. | In memory, cleared whenever the server restarts |
| Log lines | A shortened hash of the visitor's address (8 hex characters), the text length and the response time. Never the text. | Written to the service's log output; the hosting provider may keep it for its own retention period |
| Usage events | The kind of event (screening, upload, feedback), a hash of the visitor's address that changes every day, a coarse role group, the score and verdict, and yes/no flags (cached, borderline, masked). Never the text or the address. | Counts in memory until the server restarts; also written to the log output, and stored in a database table if the host configures one |

Requests to the scoring endpoint are limited to 64 KB.

### Uploads and masking

PDF, DOCX and text files (up to 3 MB) are read in memory to extract their text and are not written to disk. Only the file size and extension are logged. The extracted text is then handled like pasted text.

The mask option replaces emails, links, phone numbers and ID numbers (Aadhaar, PAN and US SSN formats) before the text is scored. It does **not** mask names or street addresses and is not an anonymisation guarantee.

### Third parties

- **Groq** receives the text (masked, if you chose that) and returns the verdict.
- **The hosting provider** (Render) serves the page and may keep its own request logs.

## Local adapter and command line

Scoring with `inference.py` is done on your machine. Nothing is sent anywhere except the one-time download of the base model and, if no local adapter folder exists, the adapter from Hugging Face.

## Annotation form

`labeling/label_resumes.html` runs entirely in the browser with no network requests. Answers are kept in the browser's local storage until the annotator downloads them. Annotator IDs are replaced by A1, A2 and so on before any result is published, and the raw label files are not committed.

## If you submitted real personal data to the demo

This project stores no resume text, so there is nothing for me to delete. The text was sent to Groq, so see Groq's privacy policy and contact Groq about it.
