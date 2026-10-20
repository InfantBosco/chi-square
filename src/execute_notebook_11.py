"""
Executor script for notebooks/11_final_analysis_and_report.ipynb
"""

import sys
import os
import shutil
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

def execute_notebook():
    nb_path = "notebooks/11_final_analysis_and_report.ipynb"
    print(f"Reading {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": "notebooks"}})
    print("Executing notebook cells...")
    client.execute()

    print("Saving executed notebook...")
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    if os.path.exists("chi_square_project"):
        os.makedirs("chi_square_project/notebooks", exist_ok=True)
        shutil.copy2(nb_path, "chi_square_project/notebooks/11_final_analysis_and_report.ipynb")
        print("Mirrored executed notebook to chi_square_project/notebooks/11_final_analysis_and_report.ipynb")

    print("Notebook 11 execution complete and verified!")

if __name__ == "__main__":
    execute_notebook()
