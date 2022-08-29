from __future__ import annotations 

from functools import partial
import os
from typing import Callable, Optional

from subprocess import run

from DistRDF.Profiling.ErrorHandling import check_and_raise_errors
from DistRDF.Profiling.Perf import CollectPerfData
from DistRDF.Profiling.Base import ProfilingData, Visualization

def collect_data_for_flamegraph(func:Callable, data_dir:str, perf_options:Optional[dict]):
    """
    Decorator for the distributed mapper function.
    It includes collection of profiling data and postprocessing to build flamegraphs
    """

    def inner(*args, **kwargs):

        # Start collecting data
        with CollectPerfData(data_dir, perf_options):
            result = func(*args, **kwargs)

        # Postprocessing
        result.prof_data = fold_perf_data(data_dir)
        return result
    
    return inner

class FlameGraphData(ProfilingData):
    """
    Handles and merges flamegraph data
    """

    def __init__(self, data:str) -> None:

        perf_table = [f.rsplit(" ",1) for f in data.splitlines()]
        self.table = dict(perf_table)

    def Merge(self, other:"FlameGraphData")->None:
        
        for key, value in other.table.items():
            self.table[key] = int(self.table.get(key, 0)) + int(value)

def fold_perf_data(data_dir:str)->"FlameGraphData":
    """
    Folds profiling data to pass it through
    """

    file = os.path.join(data_dir,f"proc-{os.getpid()}.perf.data")

    #Folding
    scripted = run(["perf", "script", "--no-demangle","-i",file], capture_output=True, text=True)
    check_and_raise_errors(scripted.stderr, "perf script")
    folded = run(["stackcollapse-perf.pl","--all"], input=scripted.stdout, capture_output=True, text=True)
    check_and_raise_errors(folded.stderr, "stackcollapse-perf.pl")

    return FlameGraphData(folded.stdout)

def restore_jit_annotation(callstack:str) -> str:
    return callstack.replace("[j]","_[j]")

def produce_flamegraph(prof_data:FlameGraphData, filename) -> None:
    """
    Produce a flamegraph given some folded flamegraph data
    """

    #build total flamegraph
    folded = ""
    for callstack, counts in prof_data.table.items():
        folded += f"{callstack} {counts}\n"

    filtered = run(["c++filt","-p"], input=folded, capture_output=True, text=True)
    check_and_raise_errors(filtered.stderr, "c++filt")
    filtered = restore_jit_annotation(filtered.stdout)

    with open(filename,"w") as file:
        run(["flamegraph.pl","--colors","java"], input=filtered, stdout=file, text=True)

class FlameGraph(Visualization):

    def __init__(
        self, 
        data_dir:str = "DistRDF_Data", 
        perf_options:Optional[dict] = None,
        filename:str = "flamegraph.svg"
    ) -> None:

        self._decorator = partial(collect_data_for_flamegraph, data_dir = data_dir, perf_options = perf_options)
        self._produce_flamegraph = partial(produce_flamegraph, filename = filename)

    def decorate(self, mapper:Callable):
        return self._decorator(mapper)
    
    def produce_visualization(self, data:FlameGraphData):
        return self._produce_flamegraph(data)