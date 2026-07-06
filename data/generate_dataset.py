"""
generate_dataset.py
--------------------
Generates a synthetic labeled dataset of (resume snippet -> structured verdict)
pairs for fine-tuning a small model as a resume screener.

Labels are rule-based (deterministic), not hand-written, so the dataset is
large, consistent, and reproducible. The model then learns the *pattern* of
turning free text into structured JSON, which is the actual skill being
fine-tuned — not resume-screening judgment itself.

Run:
    python generate_dataset.py
Produces:
    data/train.jsonl   (800 examples)
    data/eval.jsonl    (100 examples, held out)
"""

import json
import os
import random

random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ROLES = [
    "Backend Engineer", "Data Scientist", "ML Engineer", "Frontend Developer",
    "DevOps Engineer", "Full-Stack Developer", "AI Engineer", "Product Manager",
]

SKILLS_BY_ROLE = {
    "Backend Engineer": ["Python", "Java", "PostgreSQL", "REST APIs", "Docker", "Kubernetes", "Redis", "Go"],
    "Data Scientist": ["Python", "Pandas", "scikit-learn", "SQL", "Statistics", "Tableau", "R", "TensorFlow"],
    "ML Engineer": ["Python", "PyTorch", "TensorFlow", "MLOps", "Docker", "Kubernetes", "Airflow", "AWS SageMaker"],
    "Frontend Developer": ["JavaScript", "React", "TypeScript", "CSS", "Next.js", "Redux", "HTML", "Webpack"],
    "DevOps Engineer": ["AWS", "Terraform", "Kubernetes", "CI/CD", "Docker", "Ansible", "Linux", "Jenkins"],
    "Full-Stack Developer": ["JavaScript", "Python", "React", "Node.js", "PostgreSQL", "Docker", "AWS", "Git"],
    "AI Engineer": ["Python", "LangChain", "LangGraph", "Groq", "RAG", "FastAPI", "Prompt Engineering", "PyTorch"],
    "Product Manager": ["Roadmapping", "SQL", "A/B Testing", "Agile", "Figma", "Stakeholder Management", "Analytics", "JIRA"],
}

EXPERIENCE_PHRASES = [
    "{years} years of experience building {domain} systems",
    "{years} years working on {domain} at a mid-size company",
    "{years} years of hands-on {domain} development",
    "Recently graduated, {years} months of internship experience in {domain}",
]

DOMAINS_BY_ROLE = {
    "Backend Engineer": ["distributed backend", "microservices", "API", "database"],
    "Data Scientist": ["data analytics", "predictive modeling", "statistical", "data pipeline"],
    "ML Engineer": ["machine learning production", "model deployment", "MLOps", "ML pipeline"],
    "Frontend Developer": ["frontend", "UI/UX", "web application", "component library"],
    "DevOps Engineer": ["infrastructure", "cloud deployment", "CI/CD pipeline", "site reliability"],
    "Full-Stack Developer": ["full-stack web", "end-to-end application", "product", "startup"],
    "AI Engineer": ["LLM application", "agentic AI", "RAG pipeline", "generative AI"],
    "Product Manager": ["product strategy", "cross-functional product", "growth", "0-to-1 product"],
}


def generate_resume_snippet(role: str) -> tuple[str, dict]:
    """Generate one synthetic resume snippet plus its ground-truth verdict."""
    all_skills = SKILLS_BY_ROLE[role]
    num_matched = random.randint(1, len(all_skills))
    matched_skills = random.sample(all_skills, num_matched)

    years = random.choice([0, 1, 1, 2, 2, 3, 4, 5, 6, 8])
    domain = random.choice(DOMAINS_BY_ROLE[role])
    exp_phrase = random.choice(EXPERIENCE_PHRASES).format(
        years=years if years > 0 else random.randint(2, 6),
        domain=domain,
    )

    has_degree = random.random() > 0.15
    degree_line = "Bachelor's degree in Computer Science or related field. " if has_degree else ""

    snippet = (
        f"{degree_line}{exp_phrase}. "
        f"Skilled in {', '.join(matched_skills)}."
    )

    # --- Deterministic scoring rule (this is the "ground truth" the model learns) ---
    skill_coverage = num_matched / len(all_skills)
    score = round(skill_coverage * 60 + min(years, 6) * 6 + (5 if has_degree else 0))
    score = min(score, 100)

    if score >= 75:
        verdict = "strong_match"
    elif score >= 50:
        verdict = "moderate_match"
    else:
        verdict = "weak_match"

    missing_skills = [s for s in all_skills if s not in matched_skills][:3]

    label = {
        "role": role,
        "ats_score": score,
        "verdict": verdict,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "years_experience": years,
    }
    return snippet, label


def build_example(role: str) -> dict:
    snippet, label = generate_resume_snippet(role)
    prompt = (
        f"Screen this resume for a {role} position and return a structured verdict.\n\n"
        f"Resume: {snippet}"
    )
    completion = json.dumps(label, ensure_ascii=False)
    return {"prompt": prompt, "completion": completion}


def main():
    train_examples = []
    eval_examples = []

    for role in ROLES:
        for _ in range(100):
            train_examples.append(build_example(role))
        for _ in range(12 or 1):
            eval_examples.append(build_example(role))

    random.shuffle(train_examples)
    random.shuffle(eval_examples)

    train_path = os.path.join(SCRIPT_DIR, "train.jsonl")
    eval_path = os.path.join(SCRIPT_DIR, "eval.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_examples)} training examples to {train_path}")
    print(f"Wrote {len(eval_examples)} eval examples to {eval_path}")


if __name__ == "__main__":
    main()
