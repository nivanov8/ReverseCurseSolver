import argparse
from typing import Union, Dict, List, Optional

# from src.tasks.qa import (
#     QACopyPasteTask,
#     QACopyPasteEvaluator,
# )
from src.tasks.reverse_experiments.evaluator import ReverseEvaluator


def initialize_task(task_name: str, task_type: str, **args):
    task = None
    # if task_name == "qa":
    #     task = QACopyPasteTask(**args)
    if task_name == "reverse":
        task = "reverse"

    if task is None:
        raise ValueError(f"Unknown task {task}")

    return task


def initialize_evaluator(
    task_name: str, task_type: str, **args
):
    task = initialize_task(task_name, task_type, **args)
    if task_name == "reverse":
        evaluator = ReverseEvaluator(task, **args)
    else:
        raise ValueError(f"Unknown task {task}")

    return evaluator