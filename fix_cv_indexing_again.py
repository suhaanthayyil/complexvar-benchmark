with open("scripts/evaluate/crossval_fast.py", "r") as f:
    text = f.read()

import re

# Fix indexing in LegacyComplexVarGAT
old_code = """    def _get_mutant_embedding(self, x, data):
        if hasattr(data, "mutant_index"):
            mutant_idx = data.mutant_index
            if not isinstance(mutant_idx, torch.Tensor):
                mutant_idx = torch.as_tensor(mutant_idx, dtype=torch.long, device=x.device)
            if mutant_idx.ndim == 0: mutant_idx = mutant_idx.unsqueeze(0)
            return x[mutant_idx]
        return x.mean(dim=0, keepdim=True)"""

new_code = """    def _get_mutant_embedding(self, x, data):
        if hasattr(data, "mutant_index"):
            mutant_idx = data.mutant_index
            if not isinstance(mutant_idx, torch.Tensor):
                mutant_idx = torch.as_tensor(mutant_idx, dtype=torch.long, device=x.device)
            if hasattr(data, "ptr") and data.ptr is not None:
                mutant_idx = mutant_idx + data.ptr[:-1]
            return x[mutant_idx]
        return x.mean(dim=0, keepdim=True)"""

text = text.replace(old_code, new_code)

with open("scripts/evaluate/crossval_fast.py", "w") as f:
    f.write(text)
