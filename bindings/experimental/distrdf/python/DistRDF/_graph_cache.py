from __future__ import annotations
from dataclasses import dataclass

import uuid
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    import ROOT
    from DistRDF.PythonMergeables import RDataFrameFutureResult


@dataclass(frozen=True)
class ExecutionIdentifier:
    """
    A unique identifier for the current execution of the computation graph of
    a particular RDataFrame instance. The class is hashable so it can be used
    as a key in dictionaries.

    Attributes:

    rdf_uuid: An identifier for the specific RDataFrame instance.
    graph_uuid: An identifier for the computation graph sent to the workers for
        the current execution.
    """
    rdf_uuid: uuid.UUID
    graph_uuid: uuid.UUID


_RDF_REGISTER: Dict[ExecutionIdentifier, ROOT.RDataFrame] = {}
_ACTIONS_REGISTER: Dict[ExecutionIdentifier, RDataFrameFutureResult] = {}
