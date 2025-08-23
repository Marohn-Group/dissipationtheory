# lister.py
# AUthor: John A. Marohn
# Date: 2025-07-03
# Summary: This script lists all Python files in the current directory and extracts their classes and functions.

import os
import ast
import re

# List all Python files in the current directory
#  ordered such that the file name numbers are in ascending order
#  e.g., dissipation9a.py, dissipation9b.py, etc.

python_files = sorted([f for f in os.listdir('.') if f.endswith('.py')], 
                      key=lambda x:float(re.findall("(\d+)",x)[0]) if re.search("(\d+)", x) else 0)

print("Python files in this directory:")
for fname in python_files:
    print(" -", fname)

# Parse each Python file and list its classes and functions 

for file_path in python_files:

    print("\n" + "=" * len(file_path))
    print(f"{file_path}")
    print("-" * len(file_path))

    with open(file_path, "r") as f:
        source_code = f.read()

    tree = ast.parse(source_code)

    classes = []
    functions = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)

    print("Classes:")
    for class_string in classes:
        print(f" - {class_string}")
    print("Functions:")
    for function_string in functions:
        print(f" - {function_string}")
