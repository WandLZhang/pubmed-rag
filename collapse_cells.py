import json
import sys

def collapse_medical_pipeline_cells(notebook_path):
    """
    Add metadata to collapse cells in the Medical Literature Analysis Pipeline section
    """
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Track if we're in the Medical Literature Analysis Pipeline section
    in_pipeline_section = False
    cells_modified = 0
    
    # Iterate through cells
    for i, cell in enumerate(notebook['cells']):
        # Check if this is a markdown cell that starts a section
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Check if we're entering the Medical Literature Analysis Pipeline section
            if '## Medical Literature Analysis Pipeline' in source:
                in_pipeline_section = True
                print(f"Found Medical Literature Analysis Pipeline section at cell {i}")
                continue
            
            # Check if we're exiting the section (next major section)
            elif in_pipeline_section and '## Complete Example: Analyzing a Medical Case' in source:
                in_pipeline_section = False
                print(f"Exiting Medical Literature Analysis Pipeline section at cell {i}")
                break
        
        # If we're in the pipeline section, add metadata to collapse the cell
        if in_pipeline_section:
            # Initialize metadata if it doesn't exist
            if 'metadata' not in cell:
                cell['metadata'] = {}
            
            # For code cells, set the collapsed and jupyter.source_hidden properties
            if cell['cell_type'] == 'code':
                cell['metadata']['collapsed'] = True
                if 'jupyter' not in cell['metadata']:
                    cell['metadata']['jupyter'] = {}
                cell['metadata']['jupyter']['source_hidden'] = True
                cells_modified += 1
            
            # For markdown cells in subsections, we can also add tags
            elif cell['cell_type'] == 'markdown':
                source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                # Only collapse subsection content, not headers
                if not source.strip().startswith('#'):
                    if 'tags' not in cell['metadata']:
                        cell['metadata']['tags'] = []
                    if 'hide-cell' not in cell['metadata']['tags']:
                        cell['metadata']['tags'].append('hide-cell')
                    cells_modified += 1
    
    print(f"Modified {cells_modified} cells")
    
    # Write the modified notebook back to the original file
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Updated notebook: {notebook_path}")
    return notebook_path

if __name__ == "__main__":
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    else:
        notebook_path = "PubMed_RAG_Example.ipynb"
    
    collapse_medical_pipeline_cells(notebook_path)
