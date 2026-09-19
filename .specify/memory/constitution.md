# Musubi Tuner Constitution

## Core Principles

### I. Language

All Spec Kit documentation MUST be in English, including this constitution, specifications,
plans, tasks, analyses, and converge results. User dialogue, explanations, and approval
requests MUST be in Russian.

### II. Task Fidelity

The spec MUST record the original task, requirements, and scope. Agents MUST NOT change their
meaning or add their own requirements. Plans and tasks MUST follow the spec; code and tests
MUST be checked against its requirements. Requirements and expected test results MUST NOT
be adjusted to fit the implementation.

### III. Minimal, Compatible Changes

Agents MUST use existing mechanisms first and implement only what is needed for the agreed
result. New abstractions and shared interface changes MUST have a concrete justification
within the task. Agents MUST NOT rewrite a subsystem for a small feature or build generic
systems for hypothetical future needs.

Existing commands, configurations, defaults, and formats MUST be preserved; disabling a new
feature MUST leave previous behavior unchanged. Incompatible changes require explicit
authorization in the user's task; existing authorization MUST NOT be requested again.
Without that authorization, the agent MUST choose a compatible implementation. Support for
other models or modes MUST NOT expand without a separate task.

### IV. Training Invariants

Auxiliary features MUST preserve training loss, noise, data, precision, gradients, optimizer,
scheduler, RNG, and resume behavior. Validation MUST NOT update weights, replace training
loss, or affect subsequent training. A successful run alone does not prove algorithmic
correctness.

### V. Memory and Performance

Changes MUST NOT duplicate weights or retain extra tensors unnecessarily. Benchmarks and
memory or performance budgets require a separate task. Local CPU test results MUST NOT be
presented as real-model performance measurements.

### VI. Configuration and Errors

The effective configuration MUST be validated after defaults, TOML, and CLI inputs are
applied. Errors detectable without the model MUST be reported before model loading.
Unknown parameters MUST NOT be ignored. Error messages MUST identify the source, cause,
and correction.

### VII. Verification and Documentation

Changed behavior MUST be checked with suitable existing tests. New tests MUST target
substantial logic or regressions, not coverage for its own sake. Unavailable or unperformed
checks MUST NOT count as passed. Before completion, README MUST be checked against the
requirements and code and updated within scope where needed. Defects MUST NOT be concealed
by documentation changes or weakened requirements.

## Execution Stages

### Local Stage

The agent modifies code and performs available local checks. A full tool run is unavailable
in the local environment. Training, GPU runs, weight downloads, and packaging are outside
this stage and MUST NOT be launched by the agent during local development. Remote transfer
and operational verification MUST be performed as a separate stage when included in the
user's task.

The absence of external runs MUST be recorded once in the stage's results. Local checks
MUST NOT be presented as full verification of the real model.

### Remote Verification

When the user's task includes remote verification, the agent MUST proceed to that stage
after the required local checks pass, without requesting another approval. Within the
authorized task, the agent MUST transfer the implemented version to the supplied server,
prepare its environment and configurations, run the required operational checks, and
record evidence. Runs MUST serve the specified verification scenarios.

The agent MUST preserve user data and keep local and remote versions consistent after
fixes. Unavailable access, models, data, or infrastructure MUST be reported as concrete
blockers, not successful verification. Remote checks MUST NOT be added to a local-only task.

## Development Workflow

The standard sequence is `specify -> plan -> tasks -> analyze -> implement -> converge`.

The user's task authorizes the agent to complete its specified scope autonomously. The
agent MUST make routine technical decisions, generate required configurations, run the
prescribed checks, and advance between stages without human review or repeated approval.
Unspecified implementation details MUST be resolved from the requirements and code, with
material assumptions recorded. The agent MUST NOT invent unavailable external facts.

After analyze or converge, the agent MUST record a concrete list of confirmed discrepancies,
supporting evidence, proposed fixes, and verification criteria. Within the correction limit
below, it MUST correct those discrepancies autonomously in a separate editing or
implementation step. Analyze remains read-only; converge remains append-only for tasks.
Their findings MUST NOT be treated as a request for human approval.

After initial implementation, at most two autonomous `fix -> verify` rounds are allowed
per feature. A round is a correction pass over a recorded list of confirmed discrepancies
from post-implementation checks, including converge and remote verification. The coordinator
MUST record the round count and results; changing commands, agents, or verification stages
MUST NOT reset the count. A round MUST address confirmed in-scope defects; style comments,
unconfirmed hypotheses, and additional functionality MUST NOT open a correction cycle.

The agent MUST stop correction passes as soon as required checks pass and no confirmed
in-scope discrepancies remain. If discrepancies remain after two rounds, it MUST stop
further corrections and report the remaining defects, evidence, and incomplete status.
The agent MUST NOT extend the limit autonomously or weaken requirements to claim success.
Neither correction round requires separate user approval.

When multi-agent work is requested, the coordinator MUST delegate independent work, prevent
concurrent edits to the same file, integrate results, and arrange independent verification.
Quality checks and checklists MUST be assessed against evidence by the agents; human review
is not a required stage. Failed checks MUST be resolved, not marked complete to bypass them.

If progress requires unavailable access, information, or an action outside the authorized
scope, the agent MUST report the concrete blocker and remaining work, and continue any
independent authorized work. It MUST NOT declare blocked or unverified work complete.

The local stage is complete only when its requirements are met, prescribed local checks
have passed, the README review is complete, and no confirmed discrepancies remain within
the agreed scope. When remote verification is required, the whole task is complete only
after its required remote checks also pass.

## Governance

This amendment is explicitly authorized by the user. Future amendments require explicit
user authorization; authorization already provided in the current task MUST NOT be
requested again. Autonomous implementation does not authorize changing the task's goals or
weakening this constitution. Feature documents MUST NOT silently override it.

Compliance reviews MUST check feature documents and implementation against this
constitution. Conflicts within the authorized scope MUST be resolved autonomously. If a
conflict requires an unauthorized amendment or scope change, the agent MUST report that
specific blocker and continue independent authorized work. Routine decisions, stage
transitions, and confirmed-defect corrections MUST NOT be escalated for user approval.

Versions follow semantic versioning: MAJOR for incompatible principle or governance removals
or redefinitions; MINOR for new principles or sections or materially expanded guidance;
PATCH for clarifications and wording fixes without semantic changes. Dates MUST use
`YYYY-MM-DD`: Ratified retains the original adoption date; Last Amended records the date
of the latest change.

**Version**: 2.0.0 | **Ratified**: 2026-09-19 | **Last Amended**: 2026-10-05
