with open("scripts/evaluate/crossval_fast.py", "r") as f:
    text = f.read()

import re
# Fix the dummy vector size and ensuring homogeneous array
text = text.replace("all_vectors.append(np.zeros(2560 + 5)) # dummy", "all_vectors.append(np.array([]))")
text = text.replace("return np.array(all_vectors)", """
    valid_dim = next(v.shape[0] for v in all_vectors if len(v) > 0)
    for i in range(len(all_vectors)):
        if len(all_vectors[i]) == 0:
            all_vectors[i] = np.zeros(valid_dim)
    return np.array(all_vectors)""")

with open("scripts/evaluate/crossval_fast.py", "w") as f:
    f.write(text)
