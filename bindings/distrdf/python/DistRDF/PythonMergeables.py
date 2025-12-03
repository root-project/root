from __future__ import annotations

from typing import Union, List, TYPE_CHECKING

import ROOT
from ROOT._pythonization._rdataframe import AsNumpyResult

if TYPE_CHECKING:
    from DistRDF.Backends.Base import BaseBackend


class SnapshotResult(object):
    """
    Encapsulate information coming from a Snapshot operation and know how to
    merge it with other objects of this type.
    """

    def __init__(self, treename: str, filenames: List[str], resultptr: ROOT.RDF.RResultPtr = None) -> None:
        self.treename = treename
        self.filenames = filenames
        # Transient attribute, it will be discarded before the end of the mapper
        # function (in `Utils.get_mergeablevalue`) so that we don't incur in
        # serialization of the RResultPtr
        self._resultptr = resultptr

    def Merge(self, other: SnapshotResult) -> None:
        """
        When calling Snapshot on a distributed worker, a list with the path to
        the snapshotted file on the worker is stored. This function extends the
        list of the current object with the elements from the list of the other
        object.
        """
        self.filenames.extend(other.filenames)

    def GetValue(self, backend: BaseBackend):
        """
        With local RDataFrame, Snapshot returns another RDataFrame object that
        can be used to continue the application. The equivalent in the
        distributed scenario is to create a distributed RDataFrame.

        This is done by constructing a TChain with the name and the list of
        paths stored in this object. The chain is then passed to the
        `make_dataframe` function that changes depending on the backend.

        For example, if the original RDataFrame that triggered the distributed
        computation was created via a Spark backend, then this function will
        return another distributed RDataFrame build from a Spark backend
        instance. And so on for all other DistRDF backends.
        """
        snapshot_chain = ROOT.TChain(self.treename)
        # Add partial snapshot files to the chain
        for filename in self.filenames:
            snapshot_chain.Add(filename)
        # Create a new rdf with the chain and return that to user
        return backend.make_dataframe(snapshot_chain)


# A type alias to signify any type of result that can be returned from the RDataFrame API
RDataFrameFutureResult = Union[ROOT.RDF.RResultPtr, ROOT.RDF.Experimental.RResultMap, SnapshotResult, AsNumpyResult]
