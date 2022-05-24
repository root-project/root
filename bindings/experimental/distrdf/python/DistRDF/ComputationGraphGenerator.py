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
import logging

from copy import deepcopy
from functools import singledispatch
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Union

import ROOT

from DistRDF.CppWorkflow import CppWorkflow

from DistRDF.Operation import Action, AsNumpy, InstantAction, Operation, Snapshot, VariationsFor
from DistRDF.PythonMergeables import SnapshotResult

# Type hints only
if TYPE_CHECKING:
    from DistRDF.Node import Node

logger = logging.getLogger(__name__)


@singledispatch
def append_node_to_results(operation: Operation, promise: Any, results: list) -> None:
    """
    Appends the input promise to the list of results gathered while creating the
    computation graph, if the operation is an action. The promise can be of many
    types, usually a 'ROOT.RDF.RResultPtr'. Exceptions are the 'AsNumpy'
    operation which promise is an 'AsNumpyResult' and the 'Snapshot' operation
    for which a 'SnapshotResult' is created and appended to the list of results.
    """
    pass


@append_node_to_results.register(Action)
@append_node_to_results.register(InstantAction)
def _(operation: Union[Action, InstantAction], promise: Any, results: list) -> None:
    results.append(promise)


@append_node_to_results.register
def _(operation: Snapshot, promise: Any, results: list) -> None:
    results.append(SnapshotResult(operation.args[0], [operation.args[1]]))


@singledispatch
def _create_lazy_op_if_needed(operation: Operation, range_id: int) -> Operation:
    """
    We may need to change the attributes of some operations (currently
    Snapshot and AsNumpy), to make them lazy before triggering
    the computation graph. In the general case, just return the input operation.
    """
    return operation


@_create_lazy_op_if_needed.register
def _(operation: AsNumpy, range_id: int) -> AsNumpy:
    """
    The AsNumpy operation can be made lazy by setting the boolean keyword
    argument 'lazy' to 'True'.
    """
    operation.kwargs["lazy"] = True
    return operation


@_create_lazy_op_if_needed.register
def _(operation: Snapshot, range_id: int) -> Snapshot:
    """
    The Snapshot operation can be made lazy by supplying an RSnapshotOptions
    object with the 'fLazy' data member set to 'True'. Furthermore, the current
    range id needs to be appended to the input file name so that the output data
    from different tasks can be distinguished.

    Note:
    Since the file name from the original operation needs to be changed, this
    function makes a deep copy of it and returns the modified copy. This is
    needed in order to avoid that a task may receive as input an operation that
    was previously modified by another task. In that case, the file name would
    contain the range id from the other task, thus leading to create a wrong
    file name in this function.
    """
    op_modified = deepcopy(operation)

    # Retrieve filename and append range boundaries
    filename = op_modified.args[1].partition(".root")[0]
    path_with_range = "{}_{}.root".format(filename, range_id)
    # Create a partial snapshot on the current range
    op_modified.args[1] = path_with_range

    if len(op_modified.args) == 2:
        # Only the first two mandatory arguments were passed
        # Only the following overload is possible
        # Snapshot(std::string_view treename, std::string_view filename, std::string_view columnNameRegexp = "")
        op_modified.args.append("")  # Append empty regex

    if len(op_modified.args) == 4:
        # An RSnapshotOptions instance was passed as fourth argument
        # Make it lazy and keep the other options
        op_modified.args[3].fLazy = True
    else:
        # We already appended an empty regex for the 2 mandatory arguments overload
        # All other overloads have 3 mandatory arguments
        # We just need to append a lazy RSnapshotOptions now
        lazy_options = ROOT.RDF.RSnapshotOptions()
        lazy_options.fLazy = True
        op_modified.args.append(lazy_options)  # Append RSnapshotOptions

    return op_modified


@singledispatch
def _call_rdf_operation(op: Operation, parent_rdf_node: Any, range_id: int) -> Tuple[Any, Operation]:
    """
    Retrieves the concrete RDataFrame operation to be performed by
    querying the 'parent_rdf_node'. Forces lazyness on any operation, so
    they can be all chained before triggering the actual computation. Returns
    both the call to the RDataFrame operation and the operation itself, which
    are then needed when creating the list of result promises to return from
    the mapper task.
    """
    rdf_operation = getattr(parent_rdf_node, op.name)
    in_task_op = _create_lazy_op_if_needed(op, range_id)
    rdf_node = rdf_operation(*in_task_op.args, **in_task_op.kwargs)

    return rdf_node, in_task_op


@_call_rdf_operation.register
def _(op: VariationsFor, parent_rdf_node: Any, range_id: int) -> Tuple[Any, Operation]:
    """
    Implementation of a state of the computation_graph_generator
    function that is requesting systematic variations on a previously called
    action. The 'parent_rdf_node' parameter is the nominal action for which
    the variations are requested. The function calls
    ROOT.RDF.Experimental.VariationsFor on it, which returns a
    ROOT.RDF.Experimental.RResultMap. No other operations can be called on it.
    So this is the last leaf of a branch of the computation graph.
    """
    return ROOT.RDF.Experimental.VariationsFor(parent_rdf_node), op


def generate_computation_graph(graph: Dict[int, "Node"], starting_node: ROOT.RDF.RNode, range_id: int) -> List:
    """
    Generates the RDataFrame computation graph from the nodes stored in the
    input graph.

    Args:
        graph: A representation of the computation graph.
        starting_node: The RDataFrame object of this task. Specifically, if the
            head node of the computation graph is an EmptySourceHeadNode, then
            it is the result of calling the Range operation. If the head node is
            a TreeHeadNode then it is an actual RDataFrame.
        range_id: The id of the current range. Needed to assign a file name to a
            partial Snapshot if it was requested.

    Returns:
        list: List of actions of the computation graph to be triggered. Each
        element is some kind of promise of a result (usually an
        RResultPtr). Exceptions are the 'AsNumpy' operation for which an
        'AsNumpyResult' is returned and the 'Snapshot' operation for which a
        'SnapshotResult' is returned.
    """

    # Iterate over the other nodes stored in the dictionary, skipping the head
    # node. We can iterate over the values knowing that the dictionary preserves
    # the order in which it was created. Thus, we traverse the graph from top
    # to bottom, in order to create the RDF nodes in the right order.
    nodes = iter(graph.values())
    headnode = next(nodes)
    # Connect the starting node with the first node of the computation graph
    headnode.rdf_node = starting_node

    promises = []
    for node in nodes:
        rdf_node, in_task_op = _call_rdf_operation(node.operation, graph[node.parent_id].rdf_node, range_id)
        node.rdf_node = rdf_node
        append_node_to_results(in_task_op, rdf_node, promises)

    return promises


def trigger_computation_graph(graph: Dict[int, "Node"], starting_node: ROOT.RDF.RNode, range_id: int) -> List:
    """
    Trigger the computation graph.

    The list of actions to be performed is retrieved by calling
    generate_computation_graph. Afterwards, the C++ RDF computation graph is
    triggered through the `ROOT::Internal::RDF::TriggerRun` function with
    the GIL released.

    Args:
        graph: A representation of the computation graph.

        starting_node: The node where the generation of the
            computation graph is started. Either an actual RDataFrame or the
            result of a Range operation (in case of empty data source).

        range_id: The id of the current range. Needed to assign a
            file name to a partial Snapshot if it was requested.

    Returns:
        list: A list of objects that can be either used as or converted into
            mergeable values.
    """
    actions = generate_computation_graph(graph, starting_node, range_id)

    # Trigger computation graph with the GIL released
    rnode = ROOT.RDF.AsRNode(starting_node)
    ROOT.Internal.RDF.TriggerRun.__release_gil__ = True
    ROOT.Internal.RDF.TriggerRun(rnode)

    # Return a list of objects that can be later merged. In most cases this
    # is still made of RResultPtrs that will then be used as input arguments
    # to `ROOT::RDF::Detail::GetMergeableValue`. For `AsNumpy`, it returns
    # an instance of `AsNumpyResult`. For `Snapshot`, it returns a
    # `SnapshotResult`
    return actions


def run_with_cppworkflow(graph: Dict[int, "Node"], starting_node: ROOT.RDF.RNode, range_id: int) -> Tuple[List, List[str]]:
    """
    The callable that traverses the DistRDF graph nodes, generates the
    code to create the same graph in C++, compiles it and runs it.
    This function triggers the event loop via the CppWorkflow class.

    Args:
        rdf_node (ROOT.RDF.RNode): The RDataFrame node that will serve as
            the root of the computation graph.
        range_id (int): Id of the current range. Needed to assign a name
            to a partial Snapshot output file.

    Returns:
        tuple[list, list]: the first element is the list of results of the actions
            in the C++ workflow, the second element is the list of
            result types corresponding to those actions.
    """

    # Generate the code of the C++ workflow
    cpp_workflow = CppWorkflow(graph, starting_node, range_id)

    logger.debug(f"Generated C++ workflow is:\n{cpp_workflow}")

    # Compile and run the C++ workflow on the received RDF head node
    return cpp_workflow.execute()
