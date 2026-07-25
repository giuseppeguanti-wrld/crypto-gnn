---
name: ghost
description: Invoked as /ghost <filename.md> <description> to generate high-quality, human-sounding Markdown content for a target file from a natural language description. Use when the user types /ghost or asks to draft/write a Markdown document (article, note, README section, etc.) from a description of language, audience, tone, style, or constraints.
---

# Ghost

## Purpose
Generate high-quality Markdown content for a target file from a natural language description.

## Invocation
`/ghost <filename.md> <description>`

### Parameters
- `filename.md` — Destination Markdown filename. Use it only as the output filename; do not mention it inside the generated content unless explicitly requested.
- `description` — A complete description of the content to generate. This field may also contain additional instructions that define:
  - language (Italian or English);
  - audience;
  - writing style;
  - tone and register;
  - role or persona to assume;
  - formatting preferences;
  - length;
  - constraints or additional "sub-skills" to follow.

Treat every instruction inside the description as part of the specification, resolving conflicts by prioritizing the most explicit or most recent instruction.

## Behavior
- Generate only the requested Markdown document.
- Follow the description exactly while filling in reasonable details when needed.
- Structure the document using appropriate Markdown elements (headings, lists, tables, code blocks, quotes, etc.) only when they improve readability.
- Do not explain your reasoning, describe your process, or include commentary before or after the document.

## Humanized Writing Guidelines
The writing must feel genuinely human, not AI-generated.

- Write in natural, fluent Italian or English according to the request.
- Prioritize clarity, rhythm, and readability over unnecessary complexity.
- Vary sentence length and structure naturally.
- Use transitions only when they feel organic.
- Avoid repetitive paragraph patterns.
- Prefer concrete language over generic filler.
- Sound thoughtful, confident, and authentic rather than overly polished.

Avoid common AI expressions and stylistic clichés such as:
- "In conclusion", "Furthermore", "Moreover"
- "Delve into", "Embark on", "Unlock the power of"
- "Tapestry", "Landscape", "Realm"
- excessive hedging, exaggerated enthusiasm, or unnecessary formal academic language.

Do not use predictable AI writing habits such as:
- identical paragraph lengths;
- repetitive opening phrases;
- excessive bullet lists when prose is more appropriate;
- obvious "introduction-body-conclusion" templates unless explicitly requested.

## Output Format
Write the generated Markdown content to `filename.md` (creating or overwriting it as needed). Return only the generated Markdown content intended for the specified file — no preamble, no explanation, no commentary.

### Example

Input:
`/ghost article.md Write an engaging blog post in English explaining why local-first software is becoming popular. Audience: experienced developers. Tone: conversational but technical. Avoid marketing language.`

Output (written to `article.md`):
```md
# Why Local-First Software Is Finally Getting Attention

...
```
