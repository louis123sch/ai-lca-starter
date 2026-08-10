# Autonomous Iteration Protocol

## Purpose

This document captures the reusable development pattern used by the AI-LCA benchmark programme. It is intentionally domain-independent. A project can reuse this protocol while supplying its own evidence ingestion, extraction/model, evaluator, regression suite, and holdout suite.

The core loop is:

**Run -> inspect artifacts -> classify failure -> repair only if justified -> regression-test -> accept or revert -> advance -> watchdog for stalls.**

The objective is not to make a fixed benchmark set look good. The objective is to improve a general system while preserving honest evidence of generalisation.

## 1. Separate domain architecture from iteration machinery

The iteration controller must not encode paper-, dataset-, technology-, or task-specific answers.

For AI-LCA the domain architecture is:

`source evidence -> visual/native evidence ingestion -> process structure -> locked flow extraction -> human review -> Brightway matching`

Other projects may replace this pipeline while retaining the iteration protocol below.

## 2. Benchmark roles

Every benchmark must have an explicit role before its result can influence development.

### Development benchmarks

May be inspected repeatedly and may motivate general repairs. Once a benchmark influences a change, later performance on it is regression evidence, not independent generalisation evidence.

### Regression benchmarks

Protect capabilities already demonstrated. A proposed repair is not accepted if it causes a meaningful regression unless a human explicitly approves the trade-off.

### Unseen diagnostic benchmarks

Run without prior tuning. Their first result is preserved. If a genuine general defect is discovered, the benchmark may subsequently become development material, but its original unseen result remains historical evidence.

### Frozen holdouts

Used to measure generalisation. Freeze the relevant code/model/configuration SHA before running them. Do not make extraction, evaluator, prompt, fixture, gold, or threshold changes based on one holdout before the remaining holdouts in the frozen sequence have run on the identical frozen version.

Preserve original holdout metrics permanently, including poor scores and evaluator false negatives.

## 3. Immutable evaluation principles

Never:

- lower a threshold merely to obtain a pass;
- invent missing evidence, labels, LCI, expected values, or quantities;
- edit gold standards merely to match model output;
- insert benchmark-specific answers, terminology, process names, quantities, or expected structures into general extraction prompts;
- describe post-repair performance on a benchmark that influenced the repair as independent evidence of generalisation.

Evaluator false negatives should be repaired as evaluator-design problems when justified. The original metric must remain recorded.

## 4. Failure classification

Before changing model/extraction logic, classify the failure.

### A. Infrastructure failure

Examples: API timeout, connection reset, remote disconnect, runner/network failure, transient server error, rate limit.

Action: retry the failed call/job/workflow with bounded retry/backoff. Do not modify extraction logic.

### B. Evaluator-design failure

The system output is substantively supported/correct but the evaluator rejects an equivalent representation.

Action: make the smallest general evaluator improvement possible, add deterministic tests for both the intended equivalence and important non-equivalences, preserve the historical raw result, then rerun regressions.

### C. Genuine extraction/model failure

Examples: unsupported process creation, missed evidence, over-decomposition, under-decomposition, wrong attachment of flows, hallucinated quantities.

Action: inspect artifacts and source evidence; formulate the defect as a general rule; make one conservative repair; run deterministic tests and regressions.

### D. Benchmark/gold ambiguity

Source evidence does not clearly determine the expected interpretation.

Action: stop automatic repair and request human judgment. Do not silently choose the interpretation that raises the score.

## 5. Repair discipline

Repairs should be small, general, and diagnosable.

For each repair:

1. Record the failing run and artifact.
2. State the general defect without benchmark-specific answer leakage.
3. Make one conservative change where practical.
4. Run deterministic/unit tests.
5. Run the required regression suite on the same code SHA.
6. Accept only if regressions remain stable.
7. Revert promptly if the repair damages established capabilities.
8. Only then rerun the development benchmark or advance.

Use bounded automatic repair attempts. Hitting the cap should produce a diagnosable stop rather than uncontrolled prompt drift.

## 6. Same-version discipline

Comparisons are meaningful only when the relevant benchmarks run on the intended code version.

- Record the commit SHA for every benchmark run.
- For a regression gate, require all gate benchmarks to validate the same candidate SHA.
- Before dispatching dependent workflows, verify that the repository ref resolves to the intended SHA.
- For frozen holdouts, pass the frozen SHA explicitly rather than relying on the moving default branch.

## 7. Progression controller

A normal successful sequence is:

`candidate change -> deterministic tests -> regression gate -> development/unseen benchmark -> next gate`

A quality failure is not the end of autonomous work. It should normally become:

`failure -> artifact inspection -> classification -> bounded general repair -> regressions -> rerun`

Progression stops only when:

- human judgment is genuinely required;
- a repair safety cap is exhausted;
- a non-transient infrastructure blocker requires manual action;
- continuing would violate frozen-holdout isolation.

## 8. Anti-stall watchdog

Event-driven automation should do the fast work. A periodic watchdog is a backup controller.

At each watchdog check:

1. Determine whether a benchmark, repair, retry, or progression workflow is active.
2. If active, do not duplicate it.
3. If inactive, determine whether the programme is deliberately stopped.
4. If not deliberately stopped, identify the last completed state and restart the safe next action.
5. Retry infrastructure failures rather than treating them as model failures.
6. Resume bounded repair only when no equivalent repair cycle is active.
7. Rerun same-SHA regression gates when their state is incomplete or inconsistent.
8. Dispatch the next benchmark only when its gate is genuinely satisfied.

The watchdog should repair stalls, not merely report them.

## 9. Notifications

Autonomous iteration should be quiet by default. Notify a human for:

- new benchmark metrics;
- a meaningful general code/evaluator change;
- a benchmark-design ambiguity requiring judgment;
- frozen-holdout/generalisation results;
- a blocker requiring manual action.

Routine retries and healthy in-progress work need not generate notifications.

## 10. Generalisation reporting

Maintain a distinction between:

- original blind/unseen score;
- post-evaluator-correction score;
- post-development/regression score;
- frozen-holdout score.

Do not collapse these into one headline metric.

A strong claim of generalisation should rely primarily on benchmarks that did not influence the version being evaluated.

## 11. Reuse checklist

To apply this protocol to another project:

1. Define the domain pipeline.
2. Create deterministic tests for invariants.
3. Assign benchmark roles before development begins.
4. Lock gold standards and thresholds.
5. Define the regression gate.
6. Reserve multiple unseen/frozen holdouts.
7. Configure bounded infrastructure retries.
8. Configure bounded general repair.
9. Require same-SHA regression validation.
10. Add an event-driven progression controller.
11. Add a periodic anti-stall watchdog.
12. Preserve all historical metrics and candidate SHAs.
13. Freeze a release before final holdout evaluation.
14. Review failures before turning any holdout into development material.

## 12. Short invocation

For future work, the phrase **"use the autonomous iteration protocol"** means:

> Run the project using this document's benchmark-role separation, failure classification, conservative repair, same-version regression gates, frozen holdouts, infrastructure retry policy, anti-stall watchdog, historical-metric preservation, and notification rules. Domain-specific architecture and evaluation criteria remain those defined by the target project.

This protocol is a development-control method, not permission to alter evidence, gold standards, thresholds, or scientific assumptions.