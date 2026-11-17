# Security policy

## Supported versions

Only the latest release and the `main` branch receive fixes.

## Reporting a vulnerability

Please report security problems privately, not in a public issue. Use GitHub's private reporting: open the **Security** tab of this repository and choose **Report a vulnerability** (https://github.com/ayush-s-tomar/resume-screener-lora/security/advisories/new).

Please include what you found, how to reproduce it and what you think the impact is. This is a one-person student project, so I answer on a best-effort basis, usually within about a week.

## Scope

In scope: the code in this repository and the demo at https://resume-screener-finetune.onrender.com.

Out of scope: load or denial-of-service testing against the free-tier demo (it is rate limited on purpose), social engineering, and the infrastructure of Render, Groq and Hugging Face.

## Notes

- The repository contains no API keys. The demo reads its key from an environment variable on the host. If you find a secret anywhere in the git history, please report it.
- Please do not submit real resumes or personal data to the demo. Use fictional text.
