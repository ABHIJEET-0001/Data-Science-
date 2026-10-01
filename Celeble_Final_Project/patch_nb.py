import json
import os

notebook_path = 'c:/Users/Abhijeet/Data-Science--1/Celeble_Final_Project/Mini_GPT2_0.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if "def generate_text" in line:
                new_source.append("    @torch.no_grad()\n")
            new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook patched successfully!")
