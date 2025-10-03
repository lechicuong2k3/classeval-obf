# ClassEval-Obf: Code Execution Prediction Dataset with Obfuscated Identifiers

This dataset derives from [ClassEval](https://github.com/FudanSELab/ClassEval) with augmented input/output pairs for execution prediction tasks. Identifiers are obfuscated using four strategies to reduce data leakage and provide a more reliable assessment of true code execution understanding in LLMs.

## Dataset Variants

Four deterministic obfuscation strategies applied to ClassEval:

- **`classeval_obf_original.json`**: Baseline (original ClassEval with new I/O pairs)

- **`classeval_obf_alpha.json`**: **Alpha-renaming** - Role-preserving placeholders (`class1`, `method1`, `var1`)

- **`classeval_obf_ambiguity.json`**: **Ambiguous Identifiers** - Visually confusing tokens (`llllIII`, `IlllIllllIlI`)

- **`classeval_obf_crossDomain.json`**: **Cross-domain Terms** - Medical terminology (`adrenaline_fd`, `glucagon_d6`)

- **`classeval_obf_misleading.json`**: **Misleading Semantics** - Incorrect behavior names (`compute_max` for summing)

## Usage

Each JSON file contains an array of objects with:
- `code`: Python code with obfuscated identifiers
- `input`: Test input to execute
- `output`: Expected output

## Obfuscation Tool

Use `main.py` to obfuscate code by variable renaming:

```sh
python main.py --file original.py --output obfuscated.py --scheme 1 --name_length medium
```

### Configuration Options

- `--file`: Input file containing original code
- `--output`: Output file for obfuscated code
- `--name_length`: Variable name length (`short`, `medium`, `long`)
- `--scheme`: Obfuscation scheme (1-13)

## Obfuscation Schemes

The tool supports 13 different obfuscation schemes:

### Visual Confusion Schemes

**Scheme 1** - ASCII Confusables (Default)
- Maps characters to visually similar ones (a→l, b→I, c→O, etc.)
- Generates long names with underscores (e.g., `_l1O0o_l1O0o_...`)
- Length: 24-40 characters

**Scheme 2** - Ambiguous ASCII
- Uses only `l`, `I`, `1`, `O`, `0`, `o` characters
- Variable segment lengths with single/double underscores
- Creates highly ambiguous identifiers

**Scheme 3** - Unicode Homoglyphs
- Mixes Latin, Greek, and Cyrillic lookalikes
- Characters: `o`, `ο`, `О`, `о`, `c`, `с`, `e`, `е`, etc.
- Fixed 7-character segments with underscores

**Scheme 8** - Maximal Confusables
- Combines Latin, Greek, and Cyrillic lookalikes
- Includes digits `0`, `1` for non-leading positions
- Length: 28-36 characters

### Pattern-Based Schemes

**Scheme 9** - Alternating I/l Pattern (Variable Length)
- Creates patterns like `IlIlIIlIll`
- Base length: 8-20 characters
- Adds extra characters for variation

**Scheme 11** - Alternating I/l Pattern (Fixed Length)
- Maintains original identifier length
- Pure `I` and `l` alternation
- Preserves code structure visually

### Character-Based Schemes

**Scheme 6** - Leetspeak
- Replaces letters with numbers (`a→4`, `e→3`, `i→1`, etc.)
- Embeds digits at deterministic positions
- Length: 22-31 characters

**Scheme 7** - Greek Letters
- Uses Greek alphabet: `αβγδεζηθικλμνξοπρστυφχψω`
- Length: 20-31 characters

**Scheme 10** - Single Characters
- Uses single Unicode characters from multiple scripts
- ASCII, Greek, and Cyrillic letters
- Minimal identifier length

### Semantic Schemes

**Scheme 4** - Medical Dictionary
- Uses medical terminology (aorta, ventricle, alveolus, etc.)
- 1-3 words based on length setting
- Adds numeric suffixes to prevent collisions

**Scheme 5** - Meaningful Words + Role Suffix
- Combines 2-3 medical terms with misleading suffixes
- Suffixes: `manager`, `factory`, `service`, `adapter`, `provider`
- Creates semantically confusing names

**Scheme 13** - Misleading Names
- Uses predefined pools of misleading names by type:
  - Classes: `DatabaseConnection`, `HttpServer`, `JsonParser`
  - Methods: `compute_max`, `sort_desc`, `is_valid`
  - Variables: `is_empty`, `max_value`, `total_sum`

### Counter-Based Schemes

**Scheme 12** - Type-Based Counters
- Simple sequential naming: `class1`, `method1`, `var1`
- Maintains separate counters for each type
- Most straightforward obfuscation

## Name Length Control

All schemes respect the `--name_length` parameter by scaling predetermined base lengths:
- **`short`**: 40% of scheme's base length
- **`medium`**: 70% of scheme's base length  
- **`long`**: 100% of scheme's base length (default)

## Protected Identifiers

The obfuscator preserves:
- Python keywords and built-ins
- Imported module names and functions
- Special method names (`__init__`, `__str__`, etc.)
- Common parameter names (`self`, `cls`, `args`, `kwargs`)
- Regular expression methods (`match`, `search`, `findall`, etc.)
- Built-in object attributes and methods

## Example Usage

```sh
# Basic obfuscation with default scheme
python main.py --file original.py --output obfuscated.py

# Use medical terms with medium length names
python main.py --file original.py --output obfuscated.py --scheme 4 --name_length medium

# Create highly ambiguous identifiers
python main.py --file original.py --output obfuscated.py --scheme 2 --name_length long

# Generate misleading semantic names
python main.py --file original.py --output obfuscated.py --scheme 13 --name_length short
```