from PyRDF.Proxy import Proxy, ActionProxy, TransformationProxy
from PyRDF.Node import Node
from PyRDF.backend.Backend import Backend
from PyRDF import RDataFrame
import unittest
import PyRDF


class ProxyInitTest(unittest.TestCase):
    """Proxy abstract class cannot be instantiated."""

    def test_proxy_init_error(self):
        """
        Any attempt to instantiate the `Proxy` abstract class results in
        a `TypeError`.
        """
        with self.assertRaises(TypeError):
            Proxy()


class TypeReturnTest(unittest.TestCase):
    """Tests that right types are returned"""
    def test_type_return_transformation(self):
        """
        TransformationProxy object is of type `PyRDF.TransformationProxy` and
        wraps a node object.
        """
        node = Node(None, None)
        proxy = TransformationProxy(node)
        self.assertIsInstance(proxy, TransformationProxy)
        self.assertIsInstance(proxy.proxied_node, Node)

    def test_type_return_action(self):
        """
        ActionProxy object is of type `PyRDF.ActionProxy` and
        wraps a node object.
        """
        node = Node(None, None)
        proxy = ActionProxy(node)
        self.assertIsInstance(proxy, ActionProxy)
        self.assertIsInstance(proxy.proxied_node, Node)


class AttrReadTest(unittest.TestCase):
    """Test Proxy class methods."""
    class Temp(object):
        """A mock action node result class."""
        def val(self, arg):
            """A test method to check function call on the Temp class."""
            return arg + 123  # A simple operation to check

    def test_attr_simple_action(self):
        """ActionProxy object reads the right input attribute."""
        node = Node(None, None)
        proxy = ActionProxy(node)
        func = proxy.attr

        self.assertEqual(proxy._cur_attr, "attr")
        self.assertTrue(callable(func))

    def test_supported_transformation(self):
        """
        TransformationProxy object reads the right input attributes,
        returning the methods of the proxied node.
        """
        node = Node(None, None)
        proxy = TransformationProxy(node)

        transformations = {
            "Define": ["x", "tdfentry_"],
            "Filter": ["tdfentry_ > 0"],
            "Range": ["tdfentry_"]
        }

        for transformation, args in transformations.items():
            newProxy = getattr(proxy, transformation)(*args)
            self.assertEqual(proxy.proxied_node._new_op_name, transformation)
            self.assertIsInstance(newProxy, TransformationProxy)
            self.assertEqual(newProxy.proxied_node.operation.name,
                             transformation)
            self.assertEqual(newProxy.proxied_node.operation.args, args)

    def test_node_attr_transformation(self):
        """
        When a node attribute is called on a TransformationProxy object, it
        correctly returns the attribute of the proxied node.
        """
        node = Node(None, None)
        proxy = TransformationProxy(node)

        node_attributes = [
            "get_head",
            "operation",
            "children",
            "_new_op_name",
            "value",
            "pyroot_node",
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
        node = Node(None, None)
        proxy = TransformationProxy(node)
        with self.assertRaises(AttributeError):
            proxy.attribute

    def test_proxied_node_has_user_references(self):
        """
        Check that the user reference holds until the proxy lives. When the
        Python garbage collector attempts to remove the proxy object, its
        `__del__` method switches the node attribute `has_user_references` from
        `True` to `False`.
        """
        node = Node(None, None)
        proxy = TransformationProxy(node)
        self.assertTrue(node.has_user_references)
        proxy = None # noqa: avoid PEP8 F841
        self.assertFalse(node.has_user_references)

    def test_return_value(self):
        """
        Proxy object computes and returns the right output based on the
        function call.
        """
        t = AttrReadTest.Temp()
        node = Node(None, None)
        node.value = t
        proxy = ActionProxy(node)

        self.assertEqual(proxy.val(21), 144)


class GetValueTests(unittest.TestCase):
    """Check 'GetValue' instance method in Proxy."""
    class TestBackend(Backend):
        """
        Test backend to verify the working of 'GetValue' instance method
        in Proxy.
        """
        def execute(self, generator):
            """
            Test implementation of the execute method
            for 'TestBackend'. This records the head
            node of the input PyRDF graph from the
            generator object.
            """
            self.obtained_head_node = generator.head_node

        def distribute_files(self, includes_list):
            """do nothing"""
            pass

    def test_get_value_with_existing_value(self):
        """
        Test case to check the working of 'GetValue'
        method in Proxy when the current action node
        already houses a value.
        """
        node = Node(None, None)
        proxy = ActionProxy(node)
        node.value = 5

        self.assertEqual(proxy.GetValue(), 5)

    def test_get_value_with_value_not_existing(self):
        """
        Test case to check the working of 'GetValue'
        method in Proxy when the current action node
        doesn't contain a value (event-loop hasn't been
        triggered yet).
        """
        PyRDF.current_backend = GetValueTests.TestBackend()

        rdf = RDataFrame(10)  # now this is a proxy too
        count = rdf.Count()

        count.GetValue()

        # Ensure that TestBackend's execute method was called
        self.assertIs(PyRDF.current_backend.obtained_head_node,
                      rdf.proxied_node)
