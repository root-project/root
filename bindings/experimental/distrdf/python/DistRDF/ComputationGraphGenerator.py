# @author Vincenzo Eduardo Padulano
#  @author Enric Tejedor
#  @date 2021-02

################################################################################
# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import logging

from copy import deepcopy

import ROOT

from DistRDF.CppWorkflow import CppWorkflow
from DistRDF.PythonMergeables import SnapshotResult

logger = logging.getLogger(__name__)


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
            if (node_py.operation.is_action() or
                    node_py.operation.is_instant_action()):
                # Collect all action nodes in order to return them
                return_nodes.append(node_py)

        for n in node_py.children:
            # Recurse through children and collect them
            prev_nodes = self.get_action_nodes(n)

            # Attach the children nodes
            return_nodes.extend(prev_nodes)

        return return_nodes

    def _create_lazy_op_if_needed(self, operation, range_id):
        """
        We may need to change the attributes of some operations (currently
        Snapshot and AsNumpy), to make them lazy before triggering
        the computation graph.

        Note:
        For the Snapshot operation, since the file name from the original
        operation needs to be changed, this function makes a deep copy of it and
        returns the modified copy. This is needed in order to avoid that a task
        may receive as input an operation that was previously modified by
        another task. In that case, the file name would contain the range id
        from the other task, thus leading to create a wrong file name in this
        function.
        """

        if operation.name == "Snapshot":
            modified_op = deepcopy(operation)
            # Retrieve filename and append range boundaries
            filename = modified_op.args[1].partition(".root")[0]
            path_with_range = "{}_{}.root".format(filename, range_id)
            # Create a partial snapshot on the current range
            modified_op.args[1] = path_with_range

            if len(modified_op.args) == 2:
                # Only the first two mandatory arguments were passed
                # Only the following overload is possible
                # Snapshot(std::string_view treename, std::string_view filename, std::string_view columnNameRegexp = "")
                modified_op.args.append("")  # Append empty regex

            if len(modified_op.args) == 4:
                # An RSnapshotOptions instance was passed as fourth argument
                # Make it lazy and keep the other options
                modified_op.args[3].fLazy = True
            else:
                # We already appended an empty regex for the 2 mandatory arguments overload
                # All other overloads have 3 mandatory arguments
                # We just need to append a lazy RSnapshotOptions now
                lazy_options = ROOT.RDF.RSnapshotOptions()
                lazy_options.fLazy = True
                modified_op.args.append(lazy_options)  # Append RSnapshotOptions

            return modified_op

        elif operation.name == "AsNumpy":
            operation.kwargs["lazy"] = True  # Make it lazy

        return operation

    def generate_computation_graph(self, previous_node, range_id, distrdf_node=None):
        """
        Generates the RDF computation graph by recursively retrieving
        information from the DistRDF nodes.

        Args:
            previous_node (Any): The node in the RDF computation graph on which
                the operation of the current recursive state is called. In the
                first recursive state, this corresponds to the RDataFrame
                object that will be processed. Specifically, if the head node
                of the computation graph is an EmptySourceHeadNode, then the
                first current node will actually be the result of a call to the
                Range operation. If the head node is a TreeHeadNode then the
                node will be an actual RDataFrame. Successive recursive states
                will receive the result of an RDF operation call
                (e.g. Histo1D, Count).
            range_id (int): The id of the current range. Needed to assign a
                file name to a partial Snapshot if it was requested.
            distrdf_node (DistRDF.Node.Node | None): The current DistRDF node in
                the computation graph. In the first recursive state this is None
                and it will be set equal to the DistRDF headnode.

        Returns:
            list: List of actions of the computation graph to be triggered. Each
            element is some kind of promise of a result (usually an
            RResultPtr). Exceptions are the 'AsNumpy' operation for which an
            'AsNumpyResult' is returned and the 'Snapshot' operation for which a
            'SnapshotResult' is returned.
        """
        future_results = []

        if distrdf_node is None:
            # In the first recursive state, just set the
            # current DistRDF node as the head node
            distrdf_node = self.headnode
        else:
            # Execute the current operation using the output of the previous
            # node
            RDFOperation = getattr(previous_node, distrdf_node.operation.name)
            in_task_op = self._create_lazy_op_if_needed(distrdf_node.operation, range_id)
            pyroot_node = RDFOperation(*in_task_op.args, **in_task_op.kwargs)

            # The result is a pyroot object which is stored together with
            # the DistRDF node. This binds the pyroot object lifetime to the
            # DistRDF node, so both nodes will be kept alive as long as there
            # is a valid reference pointing to the DistRDF node.
            distrdf_node.pyroot_node = pyroot_node

            # Set the next `previous_node` input argument to the `pyroot_node`
            # we just retrieved
            previous_node = pyroot_node

            if (in_task_op.is_action() or in_task_op.is_instant_action()):
                if in_task_op.name == "Snapshot":
                    future_results.append(SnapshotResult(in_task_op.args[0], [in_task_op.args[1]]))
                else:
                    future_results.append(pyroot_node)

        for child_node in distrdf_node.children:
            # Recurse through children and get their output
            prev_results = self.generate_computation_graph(previous_node, range_id, child_node)

            # Attach the output of the children node
            future_results.extend(prev_results)

        return future_results

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
        actions = self.generate_computation_graph(starting_node, range_id)

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
