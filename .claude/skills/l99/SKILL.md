---
name: l99
description: Invoked as /l99 <prompt/topic> to answer with maximum technical depth, first-principles reasoning, and expert-level terminology. Controls only expertise level, analytical rigor, and precision — not output format, Markdown structure, or language. Fully composable with /ghost (which owns format, tone, language, and document structure). Use when the user types /l99 or asks for an expert-level, technically rigorous, first-principles explanation.
---

# l99

## Purpose
Produce responses with maximum technical depth, analytical rigor, and domain expertise. This skill controls only the level of expertise and precision, making it fully composable with skills such as `/ghost`.

## Usable in Combination With
- `/ghost`

When combined with `/ghost`, this skill only affects the technical depth, reasoning, and terminology. Formatting, tone, language, and document structure remain the responsibility of `/ghost`.

## Invocation
`/l99 <prompt/topic>`

The parameter may include context, assumptions, constraints, frameworks, technologies, standards, or domain-specific requirements. Infer only the minimum necessary assumptions and state them when relevant.

## l99 Rules

### Expert-Level Knowledge
Assume the reader is an expert, researcher, senior engineer, or domain specialist. Never simplify concepts unless explicitly requested.

### First-Principles Reasoning
Explain why and how concepts work, deriving conclusions from underlying mechanisms rather than surface-level descriptions.

### Technical Language
Use the terminology naturally adopted by professionals in the relevant discipline. Prefer precise technical vocabulary over simplified wording.

### Analytical Rigor
When relevant, discuss architectural decisions, implementation details, trade-offs, computational complexity, scalability, failure modes, edge cases, assumptions, limitations, and practical implications.

### Precision Over Length
Depth **does not** mean verbosity.

Explain concepts completely, but use no more words than necessary.

Avoid:
- unnecessary introductions;
- repetitive explanations;
- filler sentences;
- obvious observations;
- redundant summaries.

Every paragraph should contribute new technical information. Prefer concise, information-dense explanations over long narratives.

### Output Scope
`/l99` only controls:
- expertise level;
- technical precision;
- analytical depth;
- terminology.

It does not define:
- output format;
- Markdown organization;
- file generation behavior;
- document layout;
- output language.

If invoked together with `/ghost` (or another formatting/output skill), defer entirely to that skill for format, structure, language, and file-writing behavior.

## Example

Input

```
/l99 Explain how XGBoost differs from Random Forest.
```

Expected behavior

Provide a technically rigorous explanation covering the underlying learning paradigms, optimization process, statistical implications, computational trade-offs, hyperparameter interactions, limitations, and practical consequences, while remaining concise and avoiding unnecessary exposition.
