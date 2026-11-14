# Hand-authored test set: rubric

Purpose: a small test set written outside the dataset generator (different wording and layouts) and labelled by an explicit rubric instead of the generator's scoring rule. It is a development set, not a benchmark.

Authorship: the resumes and labels were drafted with an AI assistant against this rubric. They have not been checked by professional recruiters.

## How a resume is labelled

For a resume and its target role:

1. Count how many of the role's six core skills the resume shows (an equivalent tool counts, for example Django for "a web framework").
2. Note the years of professional experience (internships and student projects do not count as years).
3. Check whether the resume is in the role's field at all.

| Label | Rule |
|---|---|
| strong_match | 5 or 6 core skills AND years at or above the role's typical |
| moderate_match | 3 or 4 core skills, OR 5 to 6 core skills with years below the typical |
| weak_match | 2 or fewer core skills, OR the resume is from a different field |

Typical years: Backend Engineer 3, Frontend Engineer 2, AI Engineer 2, Data Engineer 3, DevOps Engineer 3, Software Developer 1.

## Core skills per role

- Backend Engineer: a backend language, a web framework, a SQL database, REST or API design, Docker, Git
- Frontend Engineer: JavaScript or TypeScript, a UI framework (React, Vue, Angular), HTML and CSS, API integration, Git, automated testing
- AI Engineer: Python, a deep-learning framework, LLM or transformer work, retrieval or vector search, evaluation, model serving
- Data Engineer: Python, SQL, ETL or pipelines, a distributed or streaming engine (Spark, Kafka), an orchestrator (Airflow), a warehouse
- DevOps Engineer: Linux, Docker, Kubernetes, CI/CD, infrastructure as code (Terraform), a public cloud
- Software Developer: a general-purpose language, data structures and algorithms, SQL, Git, unit testing, a project or internship

## Design

6 roles x 6 resumes: two strong, two moderate (one with 3 to 4 skills, one with 5 to 6 skills but too few years) and two weak (one in the same field with 0 to 2 skills, one from a different field). Three layouts are mixed: prose, labelled lines and headed bullets. No resume contains a name.

## Known limits

- 36 resumes: one resume moves accuracy by 2.8 points.
- Counting skills is a crude proxy for fit, and the same assistant drafted the resumes and the labels.
- Real resumes are messier, longer and less consistent than these.