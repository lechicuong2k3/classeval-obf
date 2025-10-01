# ClassEval-Obf: Code Understanding Dataset with Obfuscated Identifiers

This dataset evaluates Large Language Models' understanding of code when meaningful identifiers are obfuscated or replaced. Each data point contains `code`, `input`, and `output` fields.

## Dataset Variants

- **`classeval_obf_original.json`**: Original ClassEval with new input/output pairs
- **`classeval_obf_alpha.json`**: Alpha Renaming - variables renamed to single letters (a, b, c, etc.)
- **`classeval_obf_ambiguity.json`**: Ambiguous Identifiers - meaningful names replaced with ambiguous terms
- **`classeval_obf_crossDomain.json`**: Cross-domain Terms - medical domain terminology used in non-medical contexts
- **`classeval_obf_misleading.json`**: Misleading Semantics - identifiers suggest incorrect functionality

## Usage

Each JSON file contains an array of objects with:
- `code`: Python code with obfuscated identifiers
- `input`: Test input to execute
- `output`: Expected output

## Purpose

Tests whether LLMs rely on identifier names for code understanding or can reason about code structure and logic independently.