import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.vqa_runtime import run_vqa_agent


def run(input, **kwargs):
    return run_vqa_agent(input, **kwargs)
