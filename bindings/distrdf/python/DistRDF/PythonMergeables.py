from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import os
import ROOT
from ROOT._pythonization._rdataframe import AsNumpyResult

if TYPE_CHECKING:
    from .Backends.Base import BaseBackend


class SnapshotResult(object):
    """
    Encapsulate information coming from a Snapshot operation and know how to
    merge it with other objects of this type.
    """

    def __init__(
        self,
        treename: str,
        filenames: List[str],
        resultptr: ROOT.RDF.RResultPtr = None,
        return_filenames: bool = False,
        merge: bool = False,
        target_file: Optional[str] = None,
    ) -> None:
        self.treename = treename
        self.filenames = filenames
        # Transient attribute, it will be discarded before the end of the mapper
        # function (in `Utils.get_mergeablevalue`) so that we don't incur in
        # serialization of the RResultPtr
        self._resultptr = resultptr
        self.return_filenames = return_filenames
        self.merge = merge
        self.target_file = target_file

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

        If `merge` is True, the partial snapshot files are merged into the
        target file using `ROOT.TFileMerger`.

        If `return_filenames` is True, the list of snapshot file paths is
        returned instead of a distributed RDataFrame.
        """
        if self.merge and self.target_file and self.filenames:
            merger = ROOT.TFileMerger(False)
            if not merger.OutputFile(self.target_file):
                raise RuntimeError(f"Could not open output file '{self.target_file}' for merging.")
            for filename in self.filenames:
                if not merger.AddFile(filename):
                    raise RuntimeError(f"Could not add file '{filename}' to merger.")
            if not merger.Merge():
                raise RuntimeError(f"Failed to merge snapshot files into '{self.target_file}'.")
            if hasattr(merger, "CloseOutputFile"):
                merger.CloseOutputFile()

            for filename in self.filenames:
                if filename != self.target_file and os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except OSError:
                        pass

            self.filenames = [self.target_file]

            if self.return_filenames:
                return list(self.filenames)

            snapshot_chain = ROOT.TChain(self.treename)
            snapshot_chain.Add(self.target_file)
            return backend.make_dataframe(snapshot_chain)

        if self.return_filenames:
            return list(self.filenames)

        snapshot_chain = ROOT.TChain(self.treename)
        # Add partial snapshot files to the chain
        for filename in self.filenames:
            snapshot_chain.Add(filename)
        # Create a new rdf with the chain and return that to user
        return backend.make_dataframe(snapshot_chain)


# A type alias to signify any type of result that can be returned from the RDataFrame API
RDataFrameFutureResult = Union[ROOT.RDF.RResultPtr, ROOT.RDF.Experimental.RResultMap, SnapshotResult, AsNumpyResult]
