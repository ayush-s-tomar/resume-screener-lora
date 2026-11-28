# Pilot protocol: do students find a resume checker useful?

Written on 2026-11-28, before any data was collected. Status: not started.

## Question

Do students who are about to apply for internships or jobs find the hosted demo of resume-screener-lora useful, and where is it wrong? This is a small usability pilot. It does not validate the model.

## Who takes part

10 to 20 classmates and juniors who are 18 or older and volunteer after a direct message from me. I do not recruit through teachers or the placement office, I offer nothing for taking part, and I do not ask anyone twice. This is an informal student project and no ethics committee is involved; if my department requires approval for surveys of students, I get it first.

## What a participant does (about 10 minutes)

1. Open the demo at https://resume-screener-finetune.onrender.com (it can take a minute to wake up).
2. Choose a sample (fictional) resume, or paste their own with the mask option ticked. Using their own resume is optional.
3. Before pressing the button, decide how well the resume fits the role. This is the first question of the survey.
4. Run it, note the score and verdict, and look at the skills lists.
5. Open `pilot/survey.html`, answer the questions, download the small answers file and send it to me.

## Data

- The survey asks for no name, email or phone number. Answers stay in the participant's browser until they download a file with a random id and send it to me.
- Raw answer files stay on my machine in `pilot/responses/`, which git ignores. Only aggregate numbers are published (`results/pilot_v1.json`, `docs/PILOT_RESULTS.md`, one README paragraph). Free-text comments are never published.
- Because the files carry no name, I cannot withdraw one afterwards; a participant who changes their mind simply does not send the file.
- The demo itself handles data as described in `docs/PRIVACY.md`.

## Risks and safeguards

- **Privacy.** The demo sends the text to a third-party API (Groq). The invitation says so, offers sample resumes as an equally useful alternative, and recommends the mask option, which does not mask names or addresses.
- **Discouragement.** A low score can feel like a verdict on the person. The survey says in plain words that the score says nothing about worth or chances, and one question asks directly how the result felt.
- **No real decisions.** Nobody is asked to apply, rank or reject anyone based on the tool.
- **Pressure.** Invitations are one-to-one and voluntary.

## Success criteria (fixed before collecting)

The demo counts as useful in this pilot if all three hold on the valid responses:

1. At least 60% rate "I would use a tool like this again" 4 or 5.
2. At least 60% rate "the missing skills were useful to me" 4 or 5.
3. At most 20% say the result left them worried or discouraged.

I report the observed share and a 95% Wilson interval for each, whether or not the criteria are met.

## Analysis plan

Counts and shares for every question, the model's verdict against the student's own pre-result judgement (same, lower, higher), and the criteria above. No other tests. Results are written only when at least 10 valid responses exist (`python pilot/aggregate_pilot.py --write`).

## Stop rules

I stop and rethink the pilot if any participant reports distress, a privacy problem or a misunderstanding of what the tool is, or if more than one in five participants say the result left them discouraged.

## Reporting

Everything is reported, including a failed criterion. The README paragraph says that the sample is small and self-selected and that the comparison is the student's own expectation.
