from __future__ import annotations 

import os

class ProfilingError(Exception):
    """Exception raised when profiling utils (e.g. perf, stackcollapse, etc.) return errors"""
    pass

def check_and_raise_errors(stderr:str, command:str) -> None:

    if len(stderr)>0:
        
        # Comunicate which node is raising the error
        node_name = os.uname()[1]

        raise ProfilingError(f"{command} returned an error in node {node_name}:\n{stderr}")
        