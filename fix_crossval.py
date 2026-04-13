with open("scripts/evaluate/crossval.py", "r") as f:
    text = f.read()

text = text.replace('complex_model.eval()', 'complex_model.eval()\n    from complexvar.models.train import _device\n    device = _device()\n    complex_model.to(device)')
text = text.replace('monomer_model.eval()', 'monomer_model.eval()\n    monomer_model.to(device)')
text = text.replace('seq_model.eval()', 'seq_model.eval()\n    seq_model.to(device)')

with open("scripts/evaluate/crossval.py", "w") as f:
    f.write(text)
