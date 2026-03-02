import re

with open('cce_paper.tex', 'r') as f:
    content = f.read()

# Tables we want to shrink to 1 column:
# We look for \begin{table*} coupled with specific tabular definitions.

def replace_table(match):
    full_text = match.group(0)
    # Check if this table has 2, 3 or 4 columns
    tabular_match = re.search(r'\\begin\{tabular\}\{([lcr]+)\}', full_text)
    if tabular_match:
        cols = len(tabular_match.group(1))
        if cols <= 4:
            # Change table* to table
            full_text = full_text.replace('\\begin{table*}', '\\begin{table}')
            full_text = full_text.replace('\\end{table*}', '\\end{table}')
            
            # Wrap tabular in resizebox if it isn't already
            if '\\resizebox' not in full_text:
                full_text = re.sub(
                    r'(\\begin\{tabular\}.*?\\end\{tabular\})', 
                    r'\\resizebox{\\columnwidth}{!}{\n\1\n}', 
                    full_text, 
                    flags=re.DOTALL
                )
    return full_text

# Find all table* environments
new_content = re.sub(r'\\begin\{table\*\}.*?\\end\{table\*\}', replace_table, content, flags=re.DOTALL)

with open('cce_paper.tex', 'w') as f:
    f.write(new_content)

print("Tables fixed.")
