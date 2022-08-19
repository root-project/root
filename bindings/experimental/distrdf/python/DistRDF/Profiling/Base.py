from abc import ABC, abstractmethod
from typing import Callable

class ProfilingData(ABC):
    """
    Base class for the collection of any type of profiling data
    """

    @abstractmethod
    def Merge(self, other:"ProfilingData"):
        """
        Profiled data must provide a method to merge two instances of the same type.
        """
        raise NotImplementedError

class Visualization():
    """
    Base class that represents visualizations for profiling data
    """

    @property
    def Decorator(self) -> Callable:
        return lambda func: func

    @property
    def Client_task(self) -> Callable:
        return lambda task_result: None