with open("src/complexvar/models/gnn.py", "r") as f:
    text = f.read()

text = text.replace('ptr = data.ptr', 'ptr = data.ptr.cpu()')

with open("src/complexvar/models/gnn.py", "w") as f:
    f.write(text)
