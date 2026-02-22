# Testing Patterns

**Analysis Date:** 2026-02-23

## Test Framework

**Runner:**
- Custom verification scripts located in `scripts/`.
- Config: None (explicitly handled in scripts).

**Assertion Library:**
- Basic Python `if` statements and `bool` returns. No formal assertion library like `pytest` or `unittest.TestCase`.

**Run Commands:**
```bash
python scripts/verify_orchestration.py   # Verify main pipeline logic
python scripts/verify_model.py           # Verify specific model loading/inference
python scripts/verify_export.py          # Verify ONNX export results
```

## Test File Organization

**Location:**
- Verification scripts are separate from source code, located in the `scripts/` directory.

**Naming:**
- `verify_*.py` (e.g., `verify_orchestration.py`, `verify_unet_export.py`).

**Structure:**
```
scripts/
├── verify_orchestration.py
├── verify_model.py
├── verify_export.py
└── ...
```

## Test Structure

**Suite Organization:**
```python
def verify_logic():
    print("--- Verifying Logic ---")
    try:
        # 1. Setup (e.g., init pipeline in mock mode)
        # 2. Action (e.g., run stage_1)
        # 3. Verification (e.g., check result count)
        if success_condition:
            print("✅ Success")
            return True
        else:
            print("❌ Failure")
            return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    if not verify_logic():
        sys.exit(1)
```

**Patterns:**
- **Setup pattern:** Initialize objects (often with `mock=True`) and generate dummy data (e.g., `np.random.rand` for frames).
- **Teardown pattern:** Not explicitly handled (relies on garbage collection).
- **Assertion pattern:** `print()` for feedback and `return True/False` for status.

## Mocking

**Framework:** Custom (built-in to implementation).

**Patterns:**
```python
# Implementation in orchestrator.py
class PotatoPipeline:
    def __init__(self, config_path, mock=False):
        self.mock = mock

    def stage_1_leaf_segmentation(self, frame):
        if self.mock:
            # Generate mock ROI
            return mock_rois
        # Real inference...
```

**What to Mock:**
- NPU Inference (Hailo-8L) is mocked to allow testing on non-hardware environments (like local development or CI).
- Camera input (simulated with random frames).

**What NOT to Mock:**
- Pipeline orchestration logic and data flow between stages.

## Fixtures and Factories

**Test Data:**
```python
dummy_frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
```

**Location:**
- Inline within verification scripts.

## Coverage

**Requirements:** None enforced.

**View Coverage:**
- No coverage tool configured.

## Test Types

**Unit Tests:**
- Not strictly implemented as unit tests, but `verify_*.py` scripts target specific components.

**Integration Tests:**
- `verify_orchestration.py` acts as an integration test for the entire pipeline logic.

**E2E Tests:**
- Not used.

## Common Patterns

**Async Testing:**
- Not applicable (code is synchronous).

**Error Testing:**
- Handled via `try...except` blocks in verification scripts to catch and report runtime failures.

---

*Testing analysis: 2026-02-23*
