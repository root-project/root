import unittest

from DistRDF import ComputationGraphGenerator, HeadNode, Proxy
from DistRDF.Backends import Base


def create_dummy_headnode(*args):
    """Create dummy head node instance needed in the test"""
    # Pass None as `npartitions`. The tests will modify this member
    # according to needs
    return HeadNode.get_headnode(None, None, *args)


class ComputationGraphGeneratorTest(unittest.TestCase):
    """
    Check mechanism to create a callable function that returns a PyROOT object
    per each DistRDF graph node. This callable takes care of the grape pruning.
    """

    class TestBackend(Base.BaseBackend):
        """Dummy backend."""

        def ProcessAndMerge(self, ranges, mapper, reducer):
            """Dummy implementation of ProcessAndMerge."""
            pass

        def distribute_unique_paths(self, includes_list):
            """
            Dummy implementation of distribute_unique_paths. Does nothing.
            """
            pass

        def make_dataframe(self, *args, **kwargs):
            """Dummy make_dataframe"""
            pass

    class Temp(object):
        """A Class for mocking RDF CPP object."""

        def __init__(self):
            """
            Creates a mock instance. Each mock method adds an unique number to
            the `ord_list` so we can check the order in which they were called.
            """
            self.ord_list = []

        def Define(self):
            """Mock Define method"""
            self.ord_list.append(1)
            return self

        def Filter(self):
            """Mock Filter method"""
            self.ord_list.append(2)
            return self

        def Count(self):
            """Mock Count method"""
            self.ord_list.append(3)
            return self

    def test_mapper_from_graph(self):
        """A simple test case to check the working of mapper."""
        # A mock RDF object
        t = ComputationGraphGeneratorTest.Temp()

        # Head node
        hn = create_dummy_headnode(1)
        hn.backend = ComputationGraphGeneratorTest.TestBackend()
        node = Proxy.TransformationProxy(hn)
        # Set of operations to build the graph
        n1 = node.Define()
        n2 = node.Filter().Filter()
        n4 = n2.Count()
        n5 = n1.Count()
        n6 = node.Filter()  # noqa: avoid PEP8 F841

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func = ComputationGraphGenerator.generate_computation_graph
        triggerables = mapper_func(graph_dict, t, 0)
        nodes = hn._get_action_nodes()

        # Required order in the list of returned values (the nodes are stored
        # in DFS order the first time they are appended to the graph)
        reqd_order = [1, 2, 2, 3, 3, 2]

        self.assertEqual(t.ord_list, reqd_order)
        self.assertListEqual(nodes, [n4.proxied_node, n5.proxied_node])
        self.assertListEqual(triggerables, [t, t])

    def test_mapper_with_pruning(self):
        """
        A test case to check that the mapper works even in the case of
        pruning.
        """
        # A mock RDF object
        t = ComputationGraphGeneratorTest.Temp()

        # Head node
        hn = create_dummy_headnode(1)
        hn.backend = ComputationGraphGeneratorTest.TestBackend()
        node = Proxy.TransformationProxy(hn)

        # Set of operations to build the graph
        n1 = node.Define()
        n2 = node.Filter().Filter()
        n4 = n2.Count()
        n5 = n1.Count()
        n6 = node.Filter()  # noqa: avoid PEP8 F841

        # Until here the graph would be:
        # [1, 2, 2, 3, 3, 2]

        # Reason for pruning (change of reference)
        n5 = n1.Filter()  # noqa: avoid PEP8 F841

        # After the change of reference, it becomes
        # [1, 2, 2, 3, 2, 2]
        # that is, the Filter is appended at the end of the list, it is fine
        # because it holds a reference to the ID of the father.

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func = ComputationGraphGenerator.generate_computation_graph
        triggerables = mapper_func(graph_dict, t, 0)
        nodes = hn._get_action_nodes()

        reqd_order = [1, 2, 2, 3, 2, 2]

        self.assertEqual(t.ord_list, reqd_order)
        self.assertListEqual(nodes, [n4.proxied_node])
        # One occurrence of 't' per action node
        self.assertListEqual(triggerables, [t])

    def test_dfs_graph_with_pruning_transformations(self):
        """
        Test case to check that transformation nodes with no children and
        no user references get pruned.

        """
        # A mock RDF object
        t = ComputationGraphGeneratorTest.Temp()

        # Head node
        hn = create_dummy_headnode(1)
        hn.backend = ComputationGraphGeneratorTest.TestBackend()
        node = Proxy.TransformationProxy(hn)

        # Graph nodes
        n1 = node.Define()
        n2 = node.Filter()
        n3 = n2.Filter()
        n4 = n3.Count()  # noqa: avoid PEP8 F841
        n5 = n1.Filter()  # noqa: avoid PEP8 F841
        n6 = node.Filter()  # noqa: avoid PEP8 F841

        # Transformation pruning, n5 was earlier a transformation node
        n5 = n1.Count()  # noqa: avoid PEP8 F841

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func = ComputationGraphGenerator.generate_computation_graph
        triggerables = mapper_func(graph_dict, t, 0)
        nodes = hn._get_action_nodes()

        reqd_order = [1, 2, 2, 3, 2, 3]

        self.assertEqual(t.ord_list, reqd_order)
        self.assertListEqual(nodes, [n4.proxied_node, n5.proxied_node])
        # One occurrence of 't' per action node
        self.assertListEqual(triggerables, [t, t])

    def test_dfs_graph_with_recursive_pruning(self):
        """
        Test case to check that nodes in a DistRDF graph with no user references
        and no children get pruned recursively.
        """
        # A mock RDF object
        t = ComputationGraphGeneratorTest.Temp()

        # Head node
        hn = create_dummy_headnode(1)
        hn.backend = ComputationGraphGeneratorTest.TestBackend()
        node = Proxy.TransformationProxy(hn)

        # Graph nodes
        n1 = node.Define()
        n2 = node.Filter()
        n3 = n2.Filter()
        n4 = n3.Count()  # noqa: avoid PEP8 F841
        n5 = n1.Filter()  # noqa: avoid PEP8 F841
        n6 = node.Filter()  # noqa: avoid PEP8 F841

        # Remove user references from n4, n3, n2
        n4 = n3 = n2 = None  # noqa: avoid PEP8 F841

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func = ComputationGraphGenerator.generate_computation_graph
        triggerables = mapper_func(graph_dict, t, 0)
        nodes = hn._get_action_nodes()

        reqd_order = [1, 2, 2]

        self.assertEqual(t.ord_list, reqd_order)
        self.assertListEqual(nodes, [])
        # One occurrence of 't' per action node
        self.assertListEqual(triggerables, [])

    def test_dfs_graph_with_parent_pruning(self):
        """
        Test case to check that parent nodes with no user references don't
        get pruned.
        """
        # A mock RDF object
        t = ComputationGraphGeneratorTest.Temp()

        # Head node
        hn = create_dummy_headnode(1)
        hn.backend = ComputationGraphGeneratorTest.TestBackend()
        node = Proxy.TransformationProxy(hn)

        # Graph nodes
        n1 = node.Define()
        n2 = node.Filter()
        n3 = n2.Filter()
        n4 = n3.Count()  # noqa: avoid PEP8 F841
        n5 = n1.Filter()  # noqa: avoid PEP8 F841
        n6 = node.Filter()  # noqa: avoid PEP8 F841

        # Remove references from n2 (which shouldn't affect the graph)
        n2 = None

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func = ComputationGraphGenerator.generate_computation_graph
        triggerables = mapper_func(graph_dict, t, 0)
        nodes = hn._get_action_nodes()

        reqd_order = [1, 2, 2, 3, 2, 2]
        # Removing references from n2 will not prune any node
        # because n2 still has children

        self.assertEqual(t.ord_list, reqd_order)
        self.assertListEqual(nodes, [n4.proxied_node])
        # One occurrence of 't' per action node
        self.assertListEqual(triggerables, [t])

    def test_dfs_graph_with_computed_values_pruning(self):
        """
        Test case to check that computed values in action nodes get
        pruned.
        """
        # A mock RDF object
        t = ComputationGraphGeneratorTest.Temp()

        # Head node
        hn = create_dummy_headnode(1)
        hn.backend = ComputationGraphGeneratorTest.TestBackend()
        node = Proxy.TransformationProxy(hn)

        # Graph nodes
        n1 = node.Define()
        n2 = node.Filter()
        n3 = n2.Filter()
        n4 = n3.Count()  # noqa: avoid PEP8 F841
        n5 = n1.Filter()
        n6 = n5.Count()
        n7 = node.Filter()

        # This is to make sure action nodes with
        # already computed values are pruned.
        n6.proxied_node.value = 1

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func = ComputationGraphGenerator.generate_computation_graph
        triggerables = mapper_func(graph_dict, t, 0)
        nodes = hn._get_action_nodes()

        # The node 'n6' will be pruned. Hence,
        # there's only one '3' in this list.
        reqd_order = [1, 2, 2, 3, 2, 2]

        self.assertEqual(t.ord_list, reqd_order)
        self.assertListEqual(nodes, [n4.proxied_node])
        # One occurrence of 't' per action node
        self.assertListEqual(triggerables, [t])

    def test_dfs_graph_without_pruning(self):
        """
        Test case to check that node pruning does not occur if every node either
        has children or some user references.

        """
        # A mock RDF object
        t = ComputationGraphGeneratorTest.Temp()

        # Head node
        hn = create_dummy_headnode(1)
        hn.backend = ComputationGraphGeneratorTest.TestBackend()
        node = Proxy.TransformationProxy(hn)

        # Graph nodes
        n1 = node.Define()
        n2 = node.Filter()
        n3 = n2.Filter()
        n4 = n3.Count()  # noqa: avoid PEP8 F841
        n5 = n1.Count()  # noqa: avoid PEP8 F841
        n6 = node.Filter()  # noqa: avoid PEP8 F841

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func = ComputationGraphGenerator.generate_computation_graph
        triggerables = mapper_func(graph_dict, t, 0)
        nodes = hn._get_action_nodes()

        reqd_order = [1, 2, 2, 3, 3, 2]

        self.assertEqual(t.ord_list, reqd_order)
        self.assertListEqual(nodes, [n4.proxied_node, n5.proxied_node])
        # One occurrence of 't' per action node
        self.assertListEqual(triggerables, [t, t])

    def test_nodes_gt_python_recursion_limit(self):
        """
        Check that we can handle more nodes than the Python default maximum
        number of recursive function calls (1000).
        """
        # A mock RDF object
        t = ComputationGraphGeneratorTest.Temp()

        # Head node
        hn = create_dummy_headnode(1)
        hn.backend = ComputationGraphGeneratorTest.TestBackend()
        node = Proxy.TransformationProxy(hn)
        # Create three branches
        n1 = node.Define()
        n2 = node.Filter()
        # Append 1000 nodes per branch
        for _ in range(1000):
            n1 = n1.Define()
            n2 = n2.Filter()

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func = ComputationGraphGenerator.generate_computation_graph
        mapper_func(graph_dict, t, 0)

        # Required order in the list of returned values (the nodes are stored
        # in DFS order the first time they are appended to the graph)
        reqd_order = [1, 2] * (1+1000)  # (branches + 1000 nodes per branch)

        self.assertEqual(t.ord_list, reqd_order)


        # Now overwrite the branches so that we can trigger the pruning later
        n1 = node.Filter()
        n2 = node.Define()
        # Append 1000 nodes per branch
        for _ in range(1000):
            n1 = n1.Filter()
            n2 = n2.Define()
        # Reset the mock list of nodes so old nodes are not kept
        t.ord_list = []

        # Generate and execute the mapper
        graph_dict = hn._generate_graph_dict()
        mapper_func(graph_dict, t, 0)

        # Required order in the list of returned values (the nodes are stored
        # in DFS order the first time they are appended to the graph)
        reqd_order = [2, 1] * (1+1000)  # (branches + 1000 nodes per branch)

        self.assertEqual(t.ord_list, reqd_order)