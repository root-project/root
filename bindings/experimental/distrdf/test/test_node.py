import unittest

from DistRDF import Node, Proxy
from DistRDF.Backends import Base
from DistRDF.HeadNode import get_headnode


def create_dummy_headnode(*args):
    """Create dummy head node instance needed in the test"""
    # Pass None as `npartitions`. The tests will modify this member
    # according to needs
    return get_headnode(None, None, *args)


class TestBackend(Base.BaseBackend):
    """Dummy backend."""

    def ProcessAndMerge(self, ranges, mapper, reducer):
        """Dummy implementation of ProcessAndMerge."""
        pass

    def distribute_unique_paths(self, includes_list):
        """
        Dummy implementation of distribute_files. Does nothing.
        """
        pass

    def make_dataframe(self, *args, **kwargs):
        """Dummy make_dataframe"""
        pass


class OperationReadTest(unittest.TestCase):
    """
    A series of test cases to check that all new operations are created properly
    inside a new node.
    """

    def test_attr_read(self):
        """Function names are read accurately."""
        hn = create_dummy_headnode(1)
        hn.backend = TestBackend()
        node = Proxy.TransformationProxy(hn)
        func = node.Define  # noqa: avoid PEP8 F841
        self.assertEqual(node._new_op_name, "Define")

    def test_args_read(self):
        """Arguments (unnamed) are read accurately."""
        hn = create_dummy_headnode(1)
        hn.backend = TestBackend()
        node = Proxy.TransformationProxy(hn)
        newNode = node.Define(1, "b", a="1", b=2)
        self.assertEqual(newNode.operation.args, [1, "b"])

    def test_kwargs_read(self):
        """Named arguments are read accurately."""
        hn = create_dummy_headnode(1)
        hn.backend = TestBackend()
        node = Proxy.TransformationProxy(hn)
        newNode = node.Define(1, "b", a="1", b=2)
        self.assertEqual(newNode.operation.kwargs, {"a": "1", "b": 2})


class NodeReturnTest(unittest.TestCase):
    """
    A series of test cases to check that right objects are returned for a node
    (Proxy.ActionProxy, Proxy.TransformationProxy or Node).
    """

    def test_action_proxy_return(self):
        """Proxy objects are returned for action nodes."""
        hn = create_dummy_headnode(1)
        hn.backend = TestBackend()
        node = Proxy.TransformationProxy(hn)
        newNode = node.Count()
        self.assertIsInstance(newNode, Proxy.ActionProxy)
        self.assertIsInstance(newNode.proxied_node, Node.Node)

    def test_transformation_proxy_return(self):
        """Node objects are returned for transformation nodes."""
        hn = create_dummy_headnode(1)
        hn.backend = TestBackend()
        node = Proxy.TransformationProxy(hn)
        newNode = node.Define(1)
        self.assertIsInstance(newNode, Proxy.TransformationProxy)
        self.assertIsInstance(newNode.proxied_node, Node.Node)


class DunderMethodsTest(unittest.TestCase):
    """
    Test cases to check the response of the Node class for various dunder
    method calls.

    """

    def test_other_dunder_methods(self):
        """
        Test cases to check the working of other dunder methods on
        Node class.

        """
        node = Node.Node(None, None)

        # Regular dunder method must not throw an error
        node.__format__('')

        with self.assertRaises(AttributeError):
            node.__random__()  # Unknown dunder method
