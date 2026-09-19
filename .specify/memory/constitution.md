<!--
Sync Impact Report
Version change: unratified scaffold -> 1.0.0 (initial adoption).
Modified principles: none; scaffold placeholders replaced by:
  I. Language
  II. Task Fidelity
  III. Minimal, Compatible Changes
  IV. Training Invariants
  V. Memory and Performance
  VI. Configuration and Errors
  VII. Verification and Documentation
Added sections: Core Principles populated; Local Stage; Development Workflow;
  Governance populated.
Removed sections: none; illustrative template comments removed.
Deferred items / follow-up TODOs: none.
Temporary review report; remove before committing this constitution.
-->

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
feature MUST leave previous behavior unchanged. Incompatible changes require an explicit
user decision. Support for other models or modes MUST NOT expand without a separate task.

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

## Local Stage

The agent modifies code and performs available local checks. A full tool run is unavailable
in the local environment. Training, GPU runs, weight downloads, and packaging are outside
this stage and MUST NOT be launched by the agent. The tool is subsequently transferred to a
remote private server for operational verification; the agent MUST NOT perform that transfer
or verification during the local stage.

The absence of external runs MUST be recorded once in the stage's final results. Local
checks MUST NOT be presented as full verification of the real model.

## Development Workflow

The standard sequence is `specify -> plan -> tasks -> analyze -> implement -> converge`.

After converge, the agent MUST present a concrete list of confirmed discrepancies against
agreed requirements and proposed fixes. Fixes require explicit user approval of that list.
Approval applies only to that round and does not authorize the next. Finding a defect does
not expand scope; new feature requirements require a new user instruction and corresponding
feature document updates. Style comments and unconfirmed hypotheses MUST NOT open another
correction cycle.

After initial implementation, at most two `fix -> verify` rounds are allowed, each with its
own explicit approval. After the second round, the cycle MUST stop; further work requires a
new user instruction. Remaining confirmed discrepancies MUST be reported without declaring
the stage complete.

The local stage is complete only when its requirements are met, prescribed local checks
have passed, the README review is complete, and no confirmed discrepancies remain within
the agreed scope. Server verification remains a separate stage.

## Governance

This instruction authorizes initial adoption. Subsequent amendments require explicit user
permission. Feature documents MUST NOT silently override or weaken this constitution. On
conflict, the agent MUST report it and request the user's decision instead of treating a
feature document as amendment permission. Compliance reviews MUST check feature documents
and implementation against this constitution.

Versions follow semantic versioning: MAJOR for incompatible principle or governance removals
or redefinitions; MINOR for new principles or sections or materially expanded guidance;
PATCH for clarifications and wording fixes without semantic changes. Dates MUST use
`YYYY-MM-DD`: Ratified retains the original adoption date; Last Amended records the date
of the latest change.

**Version**: 1.0.0 | **Ratified**: 2026-09-19 | **Last Amended**: 2026-09-19
