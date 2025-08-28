import os
from llama_index.utils.workflow import draw_all_possible_flows


def visualize_workflow_structure_only(workflow):
    """
    Just visualize the workflow structure without executing it
    """
    output_dir = ("outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    structure_path = os.path.join(output_dir, "text2sql_workflow.html")
    print("Drawing workflow structure...")
    draw_all_possible_flows(
        workflow,
        filename=structure_path
    )
    print(f"[SUCCESS] Workflow structure saved to: {structure_path}")
