import re

with open("src/complexvar/models/train.py", "r") as f:
    content = f.read()

# Replace the _device function content
pattern = r"def _device\(\) -> str:.*?return \"cpu\""
replacement = "def _device() -> str:\n    return \"cpu\""
new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

with open("src/complexvar/models/train.py", "w") as f:
    f.write(new_content)
