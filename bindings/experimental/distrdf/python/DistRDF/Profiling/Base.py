from __future__ import annotations 

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

class Visualization:
    """
    Base class that represents visualizations for profiling data
    """

    @abstractmethod
    def decorate(self, mapper:Callable) -> Callable:
        """
        Decorate mapper function to include collection and post-processing of profiling data
        """
        raise NotImplementedError

    @abstractmethod
    def produce_visualization(self, data:ProfilingData) -> Callable:
        """
        Using post-processed data from the distributed workers, produce the visualization on the client side 
        """
        raise NotImplementedError
