#  @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
from __future__ import annotations

import logging
import os

from functools import singledispatch
from typing import Iterable, Set, Tuple

import ROOT
from ROOT._pythonization._rdataframe import AsNumpyResult

from DistRDF.PythonMergeables import SnapshotResult

logger = logging.getLogger(__name__)


def extend_include_path(include_path: str) -> None:
    """
    Extends the list of paths in which ROOT looks for headers and
    libraries. Every header directory is added to the internal include
    path of ROOT so the interpreter can find them. Even if the same path
    is added twice, ROOT keeps a collection of unique paths. Find more at
    `TInterpreter<https://root.cern.ch/doc/master/classTInterpreter.html>`_

    Args:
        include_path (str): the path to the directory containing files
            needed for the analysis.
    """
    root_path = "-I{}".format(include_path)
    ROOT.gInterpreter.AddIncludePath(root_path)

    # Retrieve ROOT internal list of include paths and add debug statement
    root_includepath = ROOT.gInterpreter.GetIncludePath()
    logger.debug("ROOT include paths:\n{}".format(root_includepath))


def declare_headers(headers_to_include: Iterable[str]) -> None:
    """
    Declares all required headers using the ROOT's C++ Interpreter.

    Args:
        headers_to_include (list): This list should consist of all
            necessary C++ headers as strings.
    """
    for header in headers_to_include:
        # Retrieve header directory
        header_dir = os.path.dirname(header)
        # Add directory to ROOT's include path
        extend_include_path(header_dir)
        # Create C++ include code
        include_code = "#include \"{}\"\n".format(header)
        try:
            ROOT.gInterpreter.Declare(include_code)
        except Exception as e:
            msg = "There was an error in including \"{}\" !".format(header)
            raise e(msg)


def declare_shared_libraries(libraries_to_include: Iterable[str]) -> None:
    """
    Declares all required shared libraries using the ROOT's C++
    Interpreter.

    Args:
        libraries_to_include (list): This list should consist of all
            necessary C++ shared libraries as strings.
    """
    for shared_library in libraries_to_include:
        # Get return value for loading the shared library.
        # On succesful load the value will be 0.
        # If the library does not exist or there was an error
        # while loading, the value will be -1
        lib_load_return = ROOT.gSystem.Load(shared_library)
        if lib_load_return == -1:
            if not os.path.exists(shared_library):
                raise IOError("Shared library does not exist!")
            raise Exception("ROOT couldn't load the shared library!")


def get_paths_set_from_string(path_string: str) -> Set[str]:
    """
    Retrieves paths to files (directory or single file) from a string.

    Args:
        path_string (str): The string to the path of the file or directory
            to be recursively searched for files.

    Returns:
        set: The set with all paths returned from the directory, or a set
            with only the path of the string.
    """
    logger.debug("Retrieving paths from {}".format(path_string))

    if os.path.isdir(path_string):
        # Create a set with all the headers in the directory
        paths_set = {
            os.path.join(rootpath, filename)
            for rootpath, dirs, filenames
            in os.walk(path_string)
            for filename
            in filenames
        }
        logger.debug("\nInitial path: {} \nPaths retrieved: {}".format(
            path_string,
            paths_set
        ))
        return paths_set
    elif os.path.isfile(path_string):
        # Convert to set if this is a string
        logger.debug("File path retrieved: {}".format(path_string))
        return {path_string}


def check_pcm_in_library_path(shared_library_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Retrieves paths to shared libraries and pcm file(s) in a directory.

    Args:
        shared_library_path (str): The string to the path of the file or
            directory to be recursively searched for files.

    Returns:
        list, list: Two lists, the first with all paths to pcm files, the
            second with all paths to shared libraries.
    """
    all_paths = get_paths_set_from_string(
        shared_library_path
    )

    pcm_paths = {
        filepath
        for filepath in all_paths
        if filepath.endswith(".pcm")
    }

    shared_library_formats = (".so", ".dll", ".dylib")
    libraries_path = {
        filepath
        for filepath in all_paths
        if filepath.endswith(shared_library_formats)
    }

    return pcm_paths, libraries_path


@singledispatch
def get_mergeablevalue(resultptr):
    """
    Generally the input argument to this function is an RResultPtr, for which a
    corresponding RMergeableValue type already exists. Call into the C++
    function to handle this case.
    """
    return ROOT.Detail.RDF.GetMergeableValue(resultptr)


@get_mergeablevalue.register(AsNumpyResult)
def _(resultptr):
    """
    Results coming from an `AsNumpy` operation can be merged with others, but
    we need to make sure to call its `GetValue` method since that will populate
    the private attribute `_py_arrays` (which is the actual dictionary of
    numpy arrays extracted from the RDataFrame columns). This extra call is an
    insurance against backends that do not automatically serialize objects
    returned by the mapper function (otherwise this would be taken care by the
    `AsNumpyResult`'s `__getstate__` method).
    """
    resultptr.GetValue()
    return resultptr


@get_mergeablevalue.register(SnapshotResult)
def _(resultptr):
    """
    When performing a distributed Snapshot we return an object holding the name
    of the dataset and the path to the partial snapshot. We can directly return
    the object, no extra work needed.
    """
    return resultptr


@singledispatch
def merge_values(mergeable_out, mergeable_in):
    """
    Generally the arguments are `RMergeableValue` instances that can be directly
    passed to the C++ function responsible for merging them.
    """
    ROOT.Detail.RDF.MergeValues(mergeable_out, mergeable_in)


@merge_values.register(AsNumpyResult)
@merge_values.register(SnapshotResult)
def _(mergeable_out, mergeable_in):
    """
    Mergeables coming from `Snapshot` or `AsNumpy` operations have their own
    `Merge` method.
    """
    mergeable_out.Merge(mergeable_in)


@singledispatch
def set_value_on_node(mergeable, node, backend):
    """
    Connects the final value after distributed computation to the corresponding
    DistRDF node.
    By default, the `GetValue` method of the mergeable returns the final value.
    """
    node.value = mergeable.GetValue()


@set_value_on_node.register
def _(mergeable: SnapshotResult, node, backend):
    """
    Connects the final value after distributed computation to the corresponding
    DistRDF node.
    This overload calls the `GetValue` method of `SnapshotResult`. This method
    accepts a 'backend' parameter because we need to recreate a distributed
    RDataFrame with the same backend of the input one.
    """
    node.value = mergeable.GetValue(backend)


@set_value_on_node.register
def _(mergeable: ROOT.Detail.RDF.RMergeableVariationsBase, node, backend):
    """
    Connects the final value after distributed computation to the corresponding
    DistRDF node.
    In this overload, the node stores the reference to the mergeable variations
    directly. It is then responsibility of the VariationsProxy object to access
    the specific varied object asked by the user, calling the right method of
    the RMergeableVariations class.
    """
    node.value = mergeable
