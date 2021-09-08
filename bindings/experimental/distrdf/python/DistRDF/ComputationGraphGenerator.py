## @author Vincenzo Eduardo Padulano
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

import ROOT
from .CppWorkflow import CppWorkflow

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

    def get_callable(self):
        """
        Converts a given graph into a callable and returns the same.

        Returns:
            function: The callable that takes in a PyROOT RDataFrame object
            and executes all operations from the DistRDF graph
            on it, recursively.
        """
        # Prune the graph to check user references
        self.headnode.graph_prune()

        def generate_computation_graph(node_cpp, range_id, node_py=None):
            """
            The callable that recurses through the DistRDF nodes and executes
            operations from a starting (PyROOT) RDF node.

            Args:
                node_cpp (ROOT.RDF.RNode): The current state's ROOT CPP node.
                    Initially this is the PyROOT RDataFrame object.
                range_id (int): The id of the current range. Needed to assign a
                    file name to a partial Snapshot if it was requested.
                node_py (optional): The current state's DistRDF node. If `None`,
                    it takes the value of `self.headnode`.

            Returns:
                list: A list of :obj:`ROOT.RResultPtr` objects in DFS order of
                their corresponding actions in the graph.
            """
            return_vals = []

            parent_node = node_cpp

            if not node_py:
                # In the first recursive state, just set the
                # current DistRDF node as the head node
                node_py = self.headnode
            else:
                # Execute the current operation using the output of the parent
                # node (node_cpp)
                RDFOperation = getattr(node_cpp, node_py.operation.name)
                operation = node_py.operation

                if operation.name == "Snapshot":
                    # Retrieve filename and append range boundaries
                    filename = operation.args[1].partition(".root")[0]
                    path_with_range = "{}_{}.root".format(filename, range_id)
                    # Create a partial snapshot on the current range
                    operation.args[1] = path_with_range
                pyroot_node = RDFOperation(*operation.args,
                                           **operation.kwargs)

                # The result is a pyroot object which is stored together with
                # the pyrdf node. This binds the pyroot object lifetime to the
                # pyrdf node, so both nodes will be kept alive as long as there
                # is a valid reference poiting to the pyrdf node.
                node_py.pyroot_node = pyroot_node

                # The new pyroot_node becomes the parent_node for the next
                # recursive call
                parent_node = pyroot_node

                if (node_py.operation.is_action() or
                        node_py.operation.is_instant_action()):
                    # Collect all action nodes in order to return them
                    # If it's a distributed snapshot return only path to
                    # the file with the partial snapshot
                    if operation.name == "Snapshot":
                        return_vals.append([path_with_range])
                    else:
                        return_vals.append(pyroot_node)

            for n in node_py.children:
                # Recurse through children and get their output
                prev_vals = generate_computation_graph(
                    parent_node, range_id, node_py=n)

                # Attach the output of the children node
                return_vals.extend(prev_vals)

            return return_vals

        return generate_computation_graph

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

            py_headnode = self.headnode
            cpp_workflow = CppWorkflow()
            parent_idx = 0

            # Recurse over children nodes
            for py_node in py_headnode.children:
                explore_graph(py_node, cpp_workflow, range_id, parent_idx)

            logger.debug("Generated C++ workflow is:\n{}".format(cpp_workflow))

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
