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

from functools import singledispatch
from typing import Any, List, Union

import ROOT

from DistRDF.CppWorkflow import CppWorkflow
from DistRDF.Node import HeadNode, Node, VariationsNode
from DistRDF.Operation import Action, AsNumpy, InstantAction, Operation, Snapshot
from DistRDF.PythonMergeables import SnapshotResult


logger = logging.getLogger(__name__)


@singledispatch
def append_node_to_actions(operation: Operation, node: Node, actions: List[Node]) -> None:
    """
    Appends the input node to the list of action nodes, if the operation of the
    node is an action.
    """
    pass


@append_node_to_actions.register(Action)
@append_node_to_actions.register(InstantAction)
def _(operation: Union[Action, InstantAction], node: Node, actions: List[Node]) -> None:
    actions.append(node)


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
def _make_op_lazy_if_needed(operation: Operation, range_id: int) -> None:
    """
    We may need to change the attributes of some operations (currently
    Snapshot and AsNumpy), to make them lazy before triggering
    the computation graph.
    """
    pass


@_make_op_lazy_if_needed.register
def _(operation: AsNumpy, range_id: int) -> None:
    operation.kwargs["lazy"] = True


@_make_op_lazy_if_needed.register
def _(operation: Snapshot, range_id: int) -> None:
    # Retrieve filename and append range boundaries
    filename = operation.args[1].partition(".root")[0]
    path_with_range = "{}_{}.root".format(filename, range_id)
    # Create a partial snapshot on the current range
    operation.args[1] = path_with_range

    if len(operation.args) == 2:
        # Only the first two mandatory arguments were passed
        # Only the following overload is possible
        # Snapshot(std::string_view treename, std::string_view filename, std::string_view columnNameRegexp = "")
        operation.args.append("")  # Append empty regex

    if len(operation.args) == 4:
        # An RSnapshotOptions instance was passed as fourth argument
        # Make it lazy and keep the other options
        operation.args[3].fLazy = True
    else:
        # We already appended an empty regex for the 2 mandatory arguments overload
        # All other overloads have 3 mandatory arguments
        # We just need to append a lazy RSnapshotOptions now
        lazy_options = ROOT.RDF.RSnapshotOptions()
        lazy_options.fLazy = True
        operation.args.append(lazy_options)  # Append RSnapshotOptions


@singledispatch
def _call_rdf_operation(distrdf_node: Node, previous_rdf_node: Any, range_id: int) -> List:
    """
    Implementation of a recursive state of the computation_graph_generator
    function. Retrieves the concrete RDataFrame operation to be performed by
    querying the 'previous_rdf_node' argument. Forces lazyness on any operation, so
    they can be all chained before triggering the actual computation. Finally,
    stores the result of calling such operation in the 'distrdf_node' argument,
    in order to bind their lifetimes and also have an input RDataFrame node for
    the next recursive state.
    """
    future_results = []

    rdf_operation = getattr(previous_rdf_node, distrdf_node.operation.name)
    _make_op_lazy_if_needed(distrdf_node.operation, range_id)
    pyroot_node = rdf_operation(*distrdf_node.operation.args, **distrdf_node.operation.kwargs)

    # The result is a PyROOT object which is stored together with
    # the DistRDF node. This binds the PyROOT object lifetime to the
    # DistRDF node, so both nodes will be kept alive as long as there
    # is a valid reference pointing to the DistRDF node.
    distrdf_node.pyroot_node = pyroot_node

    append_node_to_results(distrdf_node.operation, pyroot_node, future_results)
    return future_results


@_call_rdf_operation.register
def _(distrdf_node: HeadNode, previous_rdf_node: Any, range_id: int) -> List:
    """
    Implementation of the initial state of the computation_graph_generator
    function. The 'previous_rdf_node' parameter is some kind of ROOT::RDataFrame.
    The lifetimes of the DistRDF head node and its RDataFrame counterpart are
    bound together, in order to provide an input for the next recursive state.
    """
    distrdf_node.pyroot_node = previous_rdf_node
    return []


@_call_rdf_operation.register
def _(distrdf_node: VariationsNode, previous_rdf_node: Any, range_id: int) -> List:
    """
    Implementation of a state of the computation_graph_generator
    function that is requesting systematic variations on a previously called
    action. The 'previous_rdf_node' parameter is the nominal action for which
    the variations are requested. The function calls
    ROOT.RDF.Experimental.VariationsFor on it, which returns a
    ROOT.RDF.Experimental.RResultMap. No other operations can be called on it.
    So this is the last leaf of a branch of the computation graph.
    """
    return [ROOT.RDF.Experimental.VariationsFor(previous_rdf_node)]


def generate_computation_graph(distrdf_node: Node, previous_rdf_node: Any, range_id: int) -> List:
    """
    Generates the RDF computation graph by recursively retrieving
    information from the DistRDF nodes.

    Args:
        distrdf_node: The current DistRDF node in
            the computation graph. In the first recursive state this is None
            and it will be set equal to the DistRDF headnode.
        previous_rdf_node: The node in the RDF computation graph on which
            the operation of the current recursive state is called. In the
            first recursive state, this corresponds to the RDataFrame
            object that will be processed. Specifically, if the head node
            of the computation graph is an EmptySourceHeadNode, then the
            first current node will actually be the result of a call to the
            Range operation. If the head node is a TreeHeadNode then the
            node will be an actual RDataFrame. Successive recursive states
            will receive the result of an RDF operation call
            (e.g. Histo1D, Count).
        range_id: The id of the current range. Needed to assign a
            file name to a partial Snapshot if it was requested.

    Returns:
        list: List of actions of the computation graph to be triggered. Each
        element is some kind of promise of a result (usually an
        RResultPtr). Exceptions are the 'AsNumpy' operation for which an
        'AsNumpyResult' is returned and the 'Snapshot' operation for which a
        'SnapshotResult' is returned.
    """
    future_results = _call_rdf_operation(distrdf_node, previous_rdf_node, range_id)

    for child_node in distrdf_node.children:
        prev_results = generate_computation_graph(child_node, distrdf_node.pyroot_node, range_id)
        future_results.extend(prev_results)

    return future_results


class ComputationGraphGenerator(object):
    """
    Class that generates a callable to parse a DistRDF graph.

    Attributes:
        headnode: Head node of a DistRDF graph.
    """

    def __init__(self, headnode):
        """
        Creates a new `ComputationGraphGenerator`.

        Args:
            dataframe: DistRDF DataFrame object.
        """
        self.headnode = headnode

    def get_action_nodes(self, node_py=None):
        """
        Recurses through DistRDF graph and collects the DistRDF node objects.

        Args:
            node_py (optional): The current state's DistRDF node. If `None`, it
                takes the value of `self.headnode`.

        Returns:
            list: A list of the action nodes of the graph in DFS order, which
            coincides with the order of execution in the callable function.
        """
        return_nodes = []

        if not node_py:
            # In the first recursive state, just set the
            # current DistRDF node as the head node
            node_py = self.headnode
        else:
            append_node_to_actions(node_py.operation, node_py, return_nodes)

        for n in node_py.children:
            # Recurse through children and collect them
            prev_nodes = self.get_action_nodes(n)

            # Attach the children nodes
            return_nodes.extend(prev_nodes)

        return return_nodes

    def trigger_computation_graph(self, starting_node, range_id):
        """
        Trigger the computation graph.

        The list of actions to be performed is retrieved by calling
        generate_computation_graph. Afterwards, the C++ RDF computation graph is
        triggered through the `ROOT::Internal::RDF::TriggerRun` function with
        the GIL released.

        Args:
            starting_node (ROOT.RDF.RNode): The node where the generation of the
                computation graph is started. Either an actual RDataFrame or the
                result of a Range operation (in case of empty data source).
            range_id (int): The id of the current range. Needed to assign a
                file name to a partial Snapshot if it was requested.

        Returns:
            list: A list of objects that can be either used as or converted into
                mergeable values.
        """
        actions = generate_computation_graph(self.headnode, starting_node, range_id)

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

    def get_callable(self):
        """
        Prunes the DistRDF computation graph from unneeded nodes and returns
        a function responsible for creating and triggering the corresponding
        C++ RDF computation graph.
        """
        # Prune the graph to check user references
        # This needs to be done at this point, on the client machine, since the
        # `trigger_computation_graph` will be called inside a distributed worker.
        # Doing the pruning here makes sure we do it only once and not waste
        # extra time on the remote machines.
        self.headnode.graph_prune()

        return self.trigger_computation_graph

    def get_callable_optimized(self):
        """
        Converts a given graph into a callable and returns the same.
        The callable is optimized to execute the graph with compiled C++
        performance.

        Returns:
            function: The callable that takes in a PyROOT RDataFrame object
            and executes all operations from the DistRDF graph
            on it, recursively.
        """
        # Prune the graph to check user references
        self.headnode.graph_prune()

        def run_computation_graph(rdf_node, range_id):
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
            cpp_workflow = CppWorkflow(self.headnode, range_id)

            logger.debug("Generated C++ workflow is:\n{}".format(cpp_workflow))

            # Compile and run the C++ workflow on the received RDF head node
            return cpp_workflow.execute(ROOT.RDF.AsRNode(rdf_node))

        def explore_graph(py_node, cpp_workflow, range_id, parent_idx):
            """
            Recursively traverses the DistRDF graph nodes in DFS order and,
            for each of them, adds a new node to the C++ workflow.

            Args:
                py_node (Node): Object that contains the information to add the
                    corresponding node to the C++ workflow.
                cpp_workflow (CppWorkflow): Object that encapsulates the creation
                    of the C++ workflow graph.
                range_id (int): Id of the current range. Needed to assign a name to a
                    partial Snapshot output file.
                parent_idx (int): Index of the parent node in the C++ workflow.
            """
            node_idx = cpp_workflow.add_node(py_node.operation, range_id, parent_idx)

            for child_node in py_node.children:
                explore_graph(child_node, cpp_workflow, range_id, node_idx)

        return run_computation_graph
