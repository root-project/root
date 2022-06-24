import unittest

from DistRDF import Node
from DistRDF import Proxy
from DistRDF.Backends import Base
from DistRDF.HeadNode import get_headnode


def create_dummy_headnode(*args):
    """Create dummy head node instance needed in the test"""
    # Pass None as `npartitions`. The tests will modify this member
    # according to needs
    return get_headnode(None, None, *args)


class ProxyInitTest(unittest.TestCase):
    """Proxy abstract class cannot be instantiated."""

    def test_proxy_init_error(self):
        """
        Any attempt to instantiate the `Proxy` abstract class results in
        a `TypeError`.
        """
        with self.assertRaises(TypeError):
            Proxy.Proxy()


class TypeReturnTest(unittest.TestCase):
    """Tests that right types are returned"""

    def test_type_return_transformation(self):
        """
        TransformationProxy object is of type `DistRDF.TransformationProxy` and
        wraps a node object.
        """
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.TransformationProxy(node)
        self.assertIsInstance(proxy, Proxy.TransformationProxy)
        self.assertIsInstance(proxy.proxied_node, Node.Node)

    def test_type_return_action(self):
        """
        ActionProxy object is of type `DistRDF.ActionProxy` and
        wraps a node object.
        """
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.ActionProxy(node)
        self.assertIsInstance(proxy, Proxy.ActionProxy)
        self.assertIsInstance(proxy.proxied_node, Node.Node)


class AttrReadTest(unittest.TestCase):
    """Test Proxy class methods."""
    class Temp(object):
        """A mock action node result class."""

        def val(self, arg):
            """A test method to check function call on the Temp class."""
            return arg + 123  # A simple operation to check

    class TestBackend(Base.BaseBackend):
        """Dummy backend to test the _get_friend_info method in Dist class."""

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

        def optimize_npartitions(self):
            pass

    def test_attr_simple_action(self):
        """ActionProxy object reads the right input attribute."""
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.ActionProxy(node)
        func = proxy.attr

        self.assertEqual(proxy._cur_attr, "attr")
        self.assertTrue(callable(func))

    def test_supported_transformation(self):
        """
        TransformationProxy object reads the right input attributes,
        returning the methods of the proxied node.
        """
        node = create_dummy_headnode(1)
        node.backend = AttrReadTest.TestBackend()
        proxy = Proxy.TransformationProxy(node)

        transformations = {
            "Define": ["x", "tdfentry_"],
            "Filter": ["tdfentry_ > 0"],
        }

        for transformation, args in transformations.items():
            newProxy = getattr(proxy, transformation)(*args)
            self.assertEqual(proxy.proxied_node._new_op_name, transformation)
            self.assertIsInstance(newProxy, Proxy.TransformationProxy)
            self.assertEqual(newProxy.proxied_node.operation.name,
                             transformation)
            self.assertEqual(newProxy.proxied_node.operation.args, args)

    def test_node_attr_transformation(self):
        """
        When a node attribute is called on a TransformationProxy object, it
        correctly returns the attribute of the proxied node.
        """
        node = create_dummy_headnode(1)
        node.backend = AttrReadTest.TestBackend()
        proxy = Proxy.TransformationProxy(node)

        node_attributes = [
            "get_head",
            "operation",
            "nchildren",
            "_new_op_name",
            "value",
            "rdf_node",
            "has_user_references"
        ]

        for attr in node_attributes:
            self.assertEqual(getattr(proxy, attr),
                             getattr(proxy.proxied_node, attr))

    def test_undefined_attr_transformation(self):
        """
        When a non-defined Node class attribute is called on a
        TransformationProxy object, it raises an AttributeError.
        """
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.TransformationProxy(node)
        with self.assertRaises(AttributeError):
            proxy.attribute

    def test_proxied_node_has_user_references(self):
        """
        Check that the user reference holds until the proxy lives. When the
        Python garbage collector attempts to remove the proxy object, its
        `__del__` method switches the node attribute `has_user_references` from
        `True` to `False`.
        """
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.TransformationProxy(node)
        self.assertTrue(node.has_user_references)
        proxy = None  # noqa: avoid PEP8 F841
        self.assertFalse(node.has_user_references)

    def test_return_value(self):
        """
        Proxy object computes and returns the right output based on the
        function call.
        """
        t = AttrReadTest.Temp()
        node = create_dummy_headnode(1)
        node.backend = None
        node.value = t
        proxy = Proxy.ActionProxy(node)

        self.assertEqual(proxy.val(21), 144)


class GetValueTests(unittest.TestCase):
    """Check 'GetValue' instance method in Proxy."""
    class TestBackend(Base.BaseBackend):
        """
        Test backend to verify the working of 'GetValue' instance method
        in Proxy.
        """

        def execute(self, generator):
            """
            Test implementation of the execute method
            for 'TestBackend'. This records the head
            node of the input DistRDF graph from the
            generator object.
            """
            self.obtained_head_node = generator.head_node

        def distribute_files(self, includes_list):
            """do nothing"""
            pass

        def make_dataframe(self, *args, **kwargs):
            """Dummy make_dataframe"""
            pass

    def test_get_value_with_existing_value(self):
        """
        Test case to check the working of 'GetValue'
        method in Proxy when the current action node
        already houses a value.
        """
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.ActionProxy(node)
        node.value = 5

        self.assertEqual(proxy.GetValue(), 5)
