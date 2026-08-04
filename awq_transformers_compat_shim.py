"""
awq_transformers_compat_shim.py — compatibility shim so AutoAWQ 0.2.x imports under new transformers.

THE PROBLEM
    AutoAWQ 0.2.9 was last tested against transformers 4.51.3. Newer transformers
    (4.57+) removed activation classes that AutoAWQ imports at module load:

        from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation
        ImportError: cannot import name 'PytorchGELUTanh'

    The import fails before any model is touched, so the AWQ teacher cannot load
    even though the checkpoint and the transformers AWQ integration are both fine.

WHAT THIS DOES
    Re-creates the missing activation classes on `transformers.activations` (they
    are thin wrappers over torch's GELU variants, unchanged in behaviour), then
    imports `awq`. If a *different* symbol is missing it retries, adding each one
    it can synthesize, and reports honestly if it hits something it cannot fix.

    Only inference of an already-quantized checkpoint is needed here — nothing
    re-quantizes — so restoring these shims is behaviour-preserving for this use.

HOW TO USE
    Import it before anything that pulls in `awq` or loads an AWQ checkpoint:

        import awq_compat
        awq_compat.ensure_awq()          # returns True if `import awq` now works

    `select_data_jsd_teacher_reward.py` does this automatically if the file is present.

CAVEAT
    This is a shim, not a supported configuration. The sanctioned fix is a
    dedicated environment with transformers pinned to a version AutoAWQ supports
    (scoring and training run in separate phases, so two envs is clean). Use this
    to unblock, and note the versions in your write-up either way.
"""

import importlib
import sys

# Activation classes that AutoAWQ imports and newer transformers dropped.
# Each entry: name -> factory returning an nn.Module subclass.
_SHIMS = {}


def _build_shims():
    import torch
    import torch.nn as nn

    class PytorchGELUTanh(nn.Module):
        """transformers' old wrapper over nn.functional.gelu(approximate='tanh')."""
        def forward(self, x):
            return nn.functional.gelu(x, approximate="tanh")

    class NewGELUActivation(nn.Module):
        """GELU as used in Google BERT / OpenAI GPT (the 'new' tanh approximation)."""
        def forward(self, x):
            import math
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    class GELUActivation(nn.Module):
        def forward(self, x):
            return nn.functional.gelu(x)

    class QuickGELUActivation(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(1.702 * x)

    class ClippedGELUActivation(nn.Module):
        def __init__(self, min_val=-10.0, max_val=10.0):
            super().__init__()
            self.min_val, self.max_val = min_val, max_val

        def forward(self, x):
            return torch.clip(nn.functional.gelu(x), self.min_val, self.max_val)

    class SiLUActivation(nn.Module):
        def forward(self, x):
            return nn.functional.silu(x)

    return {
        "PytorchGELUTanh": PytorchGELUTanh,
        "NewGELUActivation": NewGELUActivation,
        "GELUActivation": GELUActivation,
        "QuickGELUActivation": QuickGELUActivation,
        "ClippedGELUActivation": ClippedGELUActivation,
        "SiLUActivation": SiLUActivation,
    }


def patch_activations(verbose=True):
    """Add any missing activation classes to transformers.activations."""
    global _SHIMS
    try:
        import transformers.activations as A
    except Exception as e:
        if verbose:
            print(f"[awq_compat] cannot import transformers.activations: {e}")
        return []
    if not _SHIMS:
        _SHIMS = _build_shims()
    added = []
    for name, cls in _SHIMS.items():
        if not hasattr(A, name):
            setattr(A, name, cls)
            added.append(name)
    if added and verbose:
        import transformers
        print(f"[awq_compat] transformers {transformers.__version__}: "
              f"restored {', '.join(added)}")
    return added


def ensure_awq(verbose=True, max_rounds=5):
    """Make `import awq` work if it reasonably can. Returns True/False."""
    for attempt in range(max_rounds):
        try:
            if "awq" in sys.modules:
                importlib.reload(sys.modules["awq"])
            else:
                importlib.import_module("awq")
            if verbose and attempt:
                print("[awq_compat] import awq succeeded after patching")
            elif verbose:
                print("[awq_compat] import awq succeeded (no patch needed)")
            return True
        except ImportError as e:
            msg = str(e)
            if verbose:
                print(f"[awq_compat] attempt {attempt + 1}: {msg}")
            # Typical: "cannot import name 'X' from 'transformers.activations'"
            if "transformers.activations" in msg or "activations" in msg:
                if not patch_activations(verbose=verbose):
                    if verbose:
                        print("[awq_compat] nothing left to patch — giving up")
                    return False
                # drop partially-imported awq modules so the retry is clean
                for m in [k for k in sys.modules if k == "awq" or k.startswith("awq.")]:
                    del sys.modules[m]
                continue
            if verbose:
                print("[awq_compat] this is not an activations problem; cannot shim it.")
                print("             Use a dedicated env with transformers pinned to a")
                print("             version AutoAWQ 0.2.9 supports (~4.51.3).")
            return False
        except Exception as e:
            if verbose:
                print(f"[awq_compat] import awq failed: {type(e).__name__}: {e}")
            return False
    if verbose:
        print(f"[awq_compat] still failing after {max_rounds} rounds")
    return False


if __name__ == "__main__":
    print("=" * 66)
    print("awq_compat self-test")
    print("=" * 66)
    try:
        import transformers
        print(f"transformers : {transformers.__version__}")
    except Exception as e:
        print(f"transformers : NOT IMPORTABLE ({e})")
    ok = ensure_awq()
    print("-" * 66)
    if ok:
        import awq
        print(f"RESULT: import awq works (version {getattr(awq, '__version__', '?')})")
        print("You can now run the JSD scoring. select_data_jsd_teacher_reward.py imports this")
        print("shim automatically when the file sits beside it.")
    else:
        print("RESULT: could not make `import awq` work.")
        print("Fall back to a dedicated scoring env:")
        print("  conda create -n awq_score --clone <your env>")
        print("  conda activate awq_score")
        print("  pip install 'transformers==4.51.3'")
        print("  python verify_tokenizer_vocab_alignment.py --student $STUDENT_BASE --teacher $TEACHER_PATH")
        print("Training keeps using the current env — the phases run separately.")
    sys.exit(0 if ok else 1)