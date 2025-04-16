from __future__ import annotations

from typing import Union, List, TYPE_CHECKING

import ROOT
import os
from ROOT._pythonization._rdataframe import AsNumpyResult

if TYPE_CHECKING:
    from DistRDF.Backends.Base import BaseBackend


class SnapshotResult(object):
    """
    Encapsulate information coming from a Snapshot operation and know how to
    merge it with other objects of this type.
    """

    MERGE_OUTPUT = True

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

        If MERGE_OUTPUT is True, it will merge all partial fils into a single ouptut file
        """
        snapshot_chain = ROOT.TChain(self.treename)
        # Add partial snapshot files to the chain
        for filename in self.filenames:
            snapshot_chain.Add(filename)

        if SnapshotResult.MERGE_OUTPUT and len(self.filenames) > 1:
            output_path = self._get_base_filename()
            merged_file_path = self._merge_snapshot_files(output_path)
            if os.path.exists(merged_file_path):
                self._cleanup_partial_files()
            merged_chain = ROOT.TChain(self.treename)
            merged_chain.Add(merged_file_path)
            self.filenames = [merged_file_path]
            return backend.make_dataframe(merged_chain)

        # Create a new rdf with the chain and return that to user
        return backend.make_dataframe(snapshot_chain)
    
    def _get_base_filename(self) -> str:
        first_file = self.filenames[0]
        if not first_file.endswith(".root"):
            return first_file
        basename = os.path.splitext(first_file)[0] # basically it will remote .root
        parts = basename.split('_')
        if parts[-1].isdigit():
            return '_'.join(parts[:-1]) + '.root'
        return first_file
    
    def _merge_snapshot_files(self, output_path: str) -> str:
        print(f"Merging {len(self.filenames)} files into {output_path}")

        chain = ROOT.TChain(self.treename)
        for filename in self.filenames:
            if os.path.exists(filename):
                print(f"Adding file: {filename}")
                chain.Add(filename)
            else:
                print(f"Warning: file {filename} does not exist")
        output_file = ROOT.TFile(output_path, "RECREATE")
        if not output_file or output_file.IsZombie():
            print(f"Error: could not create output file {output_path}")
            return ""
        print(f"Cloning tree to {output_path}")
        output_tree = Chai.CloneTree(-1, "fast")
        if not output_tree:
            print("Error: Failed to clone tree")
            #output_file.close()
            return ""
        print(f"Writing tree with {output_tree.GetEntries()} entries")
        output_tree.Write()
        output_file.Close()

        print(f"Merge Completed: {output_path}")
        return output_path
    
    def _cleanup_partial_files(self) -> None:
        base_file = self._get_base_filename()
        for filename in self.filenames:
            if filename != base_file and os.path.exists(filename):
                try:
                    os.remove(filename)
                    print(f"Removed the file {filename}")
                except Exception as e:
                    print(f"Warning: Could not remove file {filename} as {e}")


# A type alias to signify any type of result that can be returned from the RDataFrame API
RDataFrameFutureResult = Union[ROOT.RDF.RResultPtr, ROOT.RDF.Experimental.RResultMap, SnapshotResult, AsNumpyResult]
