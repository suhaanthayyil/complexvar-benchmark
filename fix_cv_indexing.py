with open("scripts/evaluate/crossval.py", "r") as f:
    text = f.read()

import re
pattern = r"if hasattr\(data, \"ptr\"\) and len\(mutant_idx\) > 1:.*?global_indices = mutant_idx"
replacement = "# Shifted automatically by torch_geometric\n            global_indices = mutant_idx"
new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

with open("scripts/evaluate/crossval.py", "w") as f:
    f.write(new_text)
