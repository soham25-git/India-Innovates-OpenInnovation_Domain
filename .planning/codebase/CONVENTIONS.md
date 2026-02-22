# Coding Conventions

**Analysis Date:** 2026-02-23

## Naming Patterns

**Files:**
- `snake_case.py`: Most Python scripts use snake_case (e.g., `orchestrator.py`, `train.py`, `prepare_calib_yolo.py`).
- `UPPERCASE.md`: Project-level documentation (e.g., `PROJECT_RULES.md`, `GSD-STYLE.md`).

**Functions:**
- `snake_case()`: Standard Python function naming (e.g., `verify_logic()`, `train_one_epoch()`).

**Variables:**
- `snake_case`: Standard variable naming (e.g., `leaf_rois`, `config_path`).

**Types:**
- `PascalCase`: Classes follow PascalCase (e.g., `PotatoPipeline`, `CFG`, `FocalLoss`).

## Code Style

**Formatting:**
- **Indentation:** 4 spaces is the standard across all `.py` files.
- **Line Length:** Generally kept within reasonable limits, though some long lines exist in training scripts.
- **Quotes:** Double quotes `"` are preferred for strings, though single quotes `'` are also used occasionally.

**Linting:**
- Not detected. No explicit linting configuration (like `.flake8` or `ruff.toml`) was found in the repository.

## Import Organization

**Order:**
1. Standard library imports (e.g., `os`, `sys`, `time`).
2. Third-party library imports (e.g., `torch`, `numpy`, `cv2`, `yaml`).
3. Local module imports (e.g., `from orchestrator import PotatoPipeline`).

**Path Aliases:**
- Not detected. Standard relative/absolute imports are used.

## Error Handling

**Patterns:**
- `try...except` blocks are used in critical initialization and verification steps (e.g., in `orchestrator.py` for `hailort` import and in `verify_orchestration.py` for logic verification).
- Errors are often logged via `print()` statements with prefixes like `❌`.

## Logging

**Framework:** `print()`
- The codebase primarily uses `print()` for logging status and errors, often with emojis for visual feedback (`🚀`, `✅`, `❌`, `⚠️`).

**Patterns:**
- `print(f"✅ Success message")`
- `print(f"❌ Error message")`

## Comments

**When to Comment:**
- Explanatory comments are used above complex logic blocks and at the start of functions.
- Sections are often marked with long comment lines (e.g., `# ==================`).

**JSDoc/TSDoc:**
- Python docstrings `"""Triple double quotes"""` are used for classes and methods.

## Function Design

**Size:**
- Functions vary from small helpers to larger training/evaluation loops (~50-100 lines).

**Parameters:**
- Uses standard positional and keyword arguments. Type hints are not consistently used.

**Return Values:**
- Standard Python return values. Verification functions often return `bool`.

## Module Design

**Exports:**
- Standard Python module exports.

**Barrel Files:**
- Not used.

---

*Convention analysis: 2026-02-23*
