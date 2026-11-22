# Flagging borderline verdicts

Computed by `selective_prediction.py` on 2026-11-23 from the saved result files, with no model run.

## Idea

The name-swap boundary test showed that verdicts close to a threshold are unstable. An application can act on that: mark a verdict **borderline** when the adapter's own score lies within a margin of a verdict boundary (49 and 74.5), and send those resumes to a human instead of deciding them automatically. An output that is not valid JSON is always flagged, because the application can detect that. The question is what the flag buys: how much accuracy rises on the resumes that are not flagged, at what cost in coverage, and whether the flag catches more errors than flagging the same number of resumes at random would.

## Rule for choosing the margin (fixed before looking at the numbers)

On the **held-out set only**, among margins of 1, 2, 3, 4, 5, 6, 8 points, take the one with the largest gain over random flagging (share of errors caught minus share of resumes flagged), provided it flags at most 30% of resumes and at least 4. If no margin has a positive gain, recommend no flag. The hand-authored set is not used to choose; it is the out-of-sample check.

## Held-out set (96 examples, 11 errors)

| Margin (points) | Flagged | Coverage | Accuracy on unflagged (95% interval) | Errors among flagged | Gain over random flagging |
|---|---|---|---|---|---|
| none | 0 (0.0%) | 100.0% | 88.5% (80.6% to 93.5%) | 0 of 11 | +0 pts |
| 1 | 3 (3.1%) | 96.9% | 89.2% (81.3% to 94.1%) | 1 of 11 | +6 pts |
| 2 | 13 (13.5%) | 86.5% | 88.0% (79.2% to 93.3%) | 1 of 11 | -4 pts |
| 3 | 15 (15.6%) | 84.4% | 88.9% (80.2% to 94.0%) | 2 of 11 | +3 pts |
| 4 | 22 (22.9%) | 77.1% | 90.5% (81.7% to 95.3%) | 4 of 11 | +13 pts |
| 5 | 22 (22.9%) | 77.1% | 90.5% (81.7% to 95.3%) | 4 of 11 | +13 pts |
| 6 | 27 (28.1%) | 71.9% | 91.3% (82.3% to 96.0%) | 5 of 11 | +17 pts |
| 8 | 55 (57.3%) | 42.7% | 92.7% (80.6% to 97.5%) | 8 of 11 | +15 pts |
| 10 | 64 (66.7%) | 33.3% | 90.6% (75.8% to 96.8%) | 8 of 11 | +6 pts |

## Hand-authored set (36 resumes, 20 errors), same margins, nothing refitted

| Margin (points) | Flagged | Coverage | Accuracy on unflagged (95% interval) | Errors among flagged | Gain over random flagging |
|---|---|---|---|---|---|
| none | 1 (2.8%) | 97.2% | 45.7% (30.5% to 61.8%) | 1 of 20 | +2 pts |
| 1 | 4 (11.1%) | 88.9% | 43.8% (28.2% to 60.7%) | 2 of 20 | -1 pts |
| 2 | 9 (25.0%) | 75.0% | 44.4% (27.6% to 62.7%) | 5 of 20 | +0 pts |
| 3 | 9 (25.0%) | 75.0% | 44.4% (27.6% to 62.7%) | 5 of 20 | +0 pts |
| 4 | 17 (47.2%) | 52.8% | 42.1% (23.1% to 63.7%) | 9 of 20 | -2 pts |
| 5 | 17 (47.2%) | 52.8% | 42.1% (23.1% to 63.7%) | 9 of 20 | -2 pts |
| 6 | 17 (47.2%) | 52.8% | 42.1% (23.1% to 63.7%) | 9 of 20 | -2 pts |
| 8 | 22 (61.1%) | 38.9% | 50.0% (26.8% to 73.2%) | 13 of 20 | +4 pts |
| 10 | 25 (69.4%) | 30.6% | 54.5% (28.0% to 78.7%) | 15 of 20 | +6 pts |

Overall accuracy without any flag: 88.5% on the held-out set and 44.4% on the hand-authored set (an invalid output counts as wrong).

## Result

**Recommended margin: 6 points.** On the held-out set it flags 27 of 96 resumes (28.1%) and those hold 5 of the 11 errors; accuracy on the unflagged resumes is 91.3% (82.3% to 96.0%) against 88.5% overall.

**Out-of-sample check.** On the hand-authored set the same margin flags 17 of 36 resumes (47.2%) and those hold 9 of the 20 errors; accuracy on the unflagged resumes is 42.1% (23.1% to 63.7%) against 44.4% overall. That is about what flagging the same share at random would catch (47.2%), so the flag does not transfer to differently written resumes.

## Limits

- The held-out set has 11 errors, so the margin is chosen from a very small number of events and could easily be different on another sample.
- Flagged resumes are not scored as right or wrong by a human; they are removed from the automatic decisions. The cost is the coverage given up.
- The boundaries (49 and 74.5) are the ones in the training data. The error analysis shows the adapter's errors on the hand-authored set are mostly under-predictions that a margin around a boundary cannot catch unless they sit near it.
- Both label sets are rubric labels, not human judgments.
