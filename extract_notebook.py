import json

# Read the notebook
with open(r'c:\Users\arbab\Desktop\myfrontend\r1-fed (1).ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

print("="*80)
print("NOTEBOOK ANALYSIS")
print("="*80)

# Extract key information
cells = notebook['cells']
print(f"\nTotal cells: {len(cells)}")
print(f"Code cells: {sum(1 for c in cells if c['cell_type'] == 'code')}")
print(f"Markdown cells: {sum(1 for c in cells if c['cell_type'] == 'markdown')}")

# Look for training results
print("\n" + "="*80)
print("SEARCHING FOR TRAINING RESULTS...")
print("="*80)

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        
        # Look for round completion or AUROC mentions
        if 'round' in source.lower() and ('auroc' in source.lower() or 'completed' in source.lower()):
            print(f"\n--- Cell {i} (Code) ---")
            print(source[:500])
            
            # Check outputs
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'text' in output:
                        text = ''.join(output['text'])
                        if 'auroc' in text.lower() or 'round' in text.lower():
                            print("\nOutput:")
                            print(text[:1000])

print("\n" + "="*80)
print("DONE")
print("="*80)
