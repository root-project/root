class CallableGenerator(object):
    """
    Class that generates a callable to parse a PyRDF graph.

    Attributes:
        head_node: Head node of a PyRDF graph.
    """
    def __init__(self, head_node):
        """
        Creates a new `CallableGenerator`.

        Args:
            head_node: Head node of a PyRDF graph.
        """
        self.head_node = head_node

    def get_action_nodes(self, node_py=None):
        """
        Recurses through PyRDF graph and collects the PyRDF node objects.

        Args:
            node_py (optional): The current state's PyRDF node. If `None`, it
                takes the value of `self.head_node`.

        Returns:
            list: A list of the action nodes of the graph in DFS order, which
            coincides with the order of execution in the callable function.
        """
        return_nodes = []

        if not node_py:
            # In the first recursive state, just set the
            # current PyRDF node as the head node
            node_py = self.head_node
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
            and executes all operations from the PyRDF graph
            on it, recursively.
        """
        # Prune the graph to check user references
        self.head_node.graph_prune()

        def mapper(node_cpp, node_py=None, rdf_range=None):
            """
            The callable that recurses through the PyRDF nodes and executes
            operations from a starting (PyROOT) RDF node.

            Args:
                node_cpp: The current state's ROOT CPP node. Initially this
                    should be given in as a PyROOT RDataFrame object.
                node_py (optional): The current state's PyRDF node. If `None`,
                    it takes the value of `self.head_node`.
                rdf_range (optional): The current range of the RDataFrame to run
                    the analysis on. This is an helper parameter for the
                    analysis in a distributed environment.

            Returns:
                list: A list of :obj:`ROOT.RResultPtr` objects in DFS order of
                their corresponding actions in the graph.
            """
            return_vals = []

            if rdf_range:
                parent_node = node_cpp.Range(rdf_range.start, rdf_range.end)
            else:
                parent_node = node_cpp

            if not node_py:
                # In the first recursive state, just set the
                # current PyRDF node as the head node
                node_py = self.head_node
            else:
                # Execute the current operation using the output of the parent
                # node (node_cpp)
                RDFOperation = getattr(node_cpp, node_py.operation.name)
                operation = node_py.operation

                if rdf_range and operation.name == "Snapshot":
                    # Retrieve filename and append range boundaries
                    filename = operation.args[1].partition(".root")[0]
                    start = str(rdf_range.start)
                    end = str(rdf_range.end - 1)
                    path_with_range = "{}_{}_{}.root".format(filename,
                                                             start, end)
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
                    if rdf_range and operation.name == "Snapshot":
                        return_vals.append([path_with_range])
                    else:
                        return_vals.append(pyroot_node)

            for n in node_py.children:
                # Recurse through children and get their output
                prev_vals = mapper(parent_node, node_py=n, rdf_range=rdf_range)

                # Attach the output of the children node
                return_vals.extend(prev_vals)

            return return_vals

        return mapper
