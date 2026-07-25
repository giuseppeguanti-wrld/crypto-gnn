---
name: role
description: Invoked as /role <role> to adapt generated content to the knowledge level and perspective of a specified audience or role (e.g. undergraduate student, graduate student, software engineer, researcher, manager). Controls only assumed background knowledge, conceptual depth, and terminology accessibility — not output format, document structure, writing style, or analytical methodology. Fully composable with /ghost, /l99, and /ooda. Use when the user types /role or asks to adapt/target content for a specific audience or reader's expertise level.
---

# role

## Purpose
Adapt the generated content to the knowledge level and perspective of a specified audience or role. This skill adjusts the assumed expertise of the reader without changing the document's format or purpose, making it fully composable with skills such as `/ghost`, `/l99`, and `/ooda`.

## Usable in Combination With
- `/ghost`
- `/l99`
- `/ooda`

When combined:
- `/ghost` defines format, language, and writing style.
- `/l99` defines technical depth and analytical rigor.
- `/ooda` defines planning logic and task decomposition.
- `/role` defines the assumed background knowledge, conceptual depth, and terminology accessibility of the target reader.

If invoked together with one or more of these skills, `/role` only recalibrates who the explanation is pitched at — it defers entirely to the other skills for format, structure, language, rigor, and methodology.

## Invocation
`/role <role>`

The parameter consists solely of the intended reader's role (e.g. undergraduate student, graduate student, software engineer, researcher, manager).

## Rules

- Assume the specified role represents the reader's background knowledge.
- Adjust the level of explanation, terminology, and conceptual abstraction accordingly.
- Preserve technical accuracy while avoiding unnecessary complexity.
- Introduce specialized concepts only when appropriate for the specified role.
- Provide sufficient context for unfamiliar concepts without becoming verbose.
- Never oversimplify or omit important technical details; instead, calibrate how they are explained.

## Scope

`/role` determines:
- assumed background knowledge;
- conceptual depth;
- terminology accessibility;
- amount of implicit knowledge.

It does **not** determine:
- output format;
- document structure;
- writing style;
- analytical methodology.

These aspects are delegated to other compatible skills.

## Example

Input

```
/role software engineer /l99 Explain how graph convolutional networks propagate information.
```

Expected behavior

Explain graph convolutional networks assuming familiarity with software engineering and general machine learning concepts (e.g. matrices, neural network layers, training loops) but without assuming a background in spectral graph theory — introducing notions like the graph Laplacian or spectral filtering with enough context to follow, rather than assuming they are already known, while `/l99` still governs the technical rigor and precision of the explanation itself.
