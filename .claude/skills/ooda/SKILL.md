---
name: ooda
description: Invoked as /ooda <description/objective> to apply the Observe-Orient-Decide-Act (OODA) framework and turn a goal, problem, or project into a structured, actionable execution roadmap. Controls only planning logic and task decomposition — not format, language, or technical depth. Fully composable with /ghost (format/language/style) and /L99 (technical depth/rigor). Use when the user types /ooda or asks for a strategic plan, roadmap, or breakdown of a project/problem into milestones and tasks.
---

# ooda

## Purpose
Apply the Observe–Orient–Decide–Act (OODA) framework to transform a goal, problem, or project into a clear execution strategy. This skill only defines the planning methodology and is designed to compose with `/ghost` and `/L99`.

## Usable in Combination With
- `/ghost`
- `/L99`

When combined:
- `/ghost` defines format, language, and writing style.
- `/L99` defines technical depth and analytical rigor.
- `/ooda` defines planning logic and task decomposition.

## Invocation
`/ooda <description/objective>`

The parameter may describe a project, technical problem, research objective, or operational challenge. Infer only the minimum necessary assumptions.

## ooda Rules

**Observe**
Identify objectives, facts, assumptions, constraints, resources, dependencies, and missing information.

**Orient**
Analyze risks, bottlenecks, trade-offs, alternative approaches, and the critical path.

**Decide**
Select the most suitable strategy, defining priorities, milestones, decision points, and success criteria.

**Act**
Produce an ordered implementation plan with concrete, actionable tasks. Minimize unnecessary complexity and rework.

## Planning Principles

- Prioritize execution over discussion.
- Make dependencies explicit.
- Validate early and often.
- Prefer measurable deliverables over generic recommendations.

## Example

```
/ooda Build an NBA game prediction pipeline.
```

Expected behavior:

Produce a structured roadmap that identifies the current state, analyzes constraints, chooses an implementation strategy, and breaks the work into sequential, actionable milestones.
