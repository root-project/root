from array import array
import unittest
import os

import ROOT

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
        NodeProxy object is of type `DistRDF.NodeProxy` and
        wraps a node object.
        """
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.NodeProxy(node)
        self.assertIsInstance(proxy, Proxy.NodeProxy)
        self.assertIsInstance(proxy.proxied_node, Node.Node)

    def test_type_return_action(self):
        """
        ResultPtrProxy object is of type `DistRDF.ResultPtrProxy` and
        wraps a node object.
        """
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.ResultPtrProxy(node)
        self.assertIsInstance(proxy, Proxy.ResultPtrProxy)
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
        """ResultPtrProxy object reads the right input attribute."""
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.ResultPtrProxy(node)
        func = proxy.attr

        self.assertEqual(proxy._cur_attr, "attr")
        self.assertTrue(callable(func))

    def test_supported_transformation(self):
        """
        NodeProxy object reads the right input attributes,
        returning the methods of the proxied node.
        """
        node = create_dummy_headnode(1)
        node.backend = AttrReadTest.TestBackend()
        proxy = Proxy.NodeProxy(node)

        transformations = {
            "Define": ["x", "1"],
            "Filter": ["x > 0"],
        }

        for transformation, args in transformations.items():
            parent_node = proxy.proxied_node
            proxy = getattr(proxy, transformation)(*args)
            # Calling the operation on the parent node modifies an attribute
            self.assertEqual(parent_node._new_op_name, transformation)
            self.assertIsInstance(proxy, Proxy.NodeProxy)
            self.assertEqual(proxy.proxied_node.operation.name,
                             transformation)
            self.assertEqual(proxy.proxied_node.operation.args, args)

    def test_node_attr_transformation(self):
        """
        When a node attribute is called on a NodeProxy object, it
        correctly returns the attribute of the proxied node.
        """
        node = create_dummy_headnode(1)
        node.backend = AttrReadTest.TestBackend()
        proxy = Proxy.NodeProxy(node)

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
        NodeProxy object, it raises an AttributeError.
        """
        node = create_dummy_headnode(1)
        node.backend = None
        proxy = Proxy.NodeProxy(node)
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
        proxy = Proxy.NodeProxy(node)
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
        proxy = Proxy.ResultPtrProxy(node)

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
        proxy = Proxy.ResultPtrProxy(node)
        node.value = 5

        self.assertEqual(proxy.GetValue(), 5)

class InternalDataFrameTests(unittest.TestCase):
    """The HeadNode stores an internal RDataFrame for certain information"""

    @classmethod
    def setUpClass(cls):
        """Create a dummy file to use for the RDataFrame constructor."""
        cls.test_treename = "treename"
        cls.test_filename = "test_distrdf_getcolumnnames.root"
        cls.test_tree_entries = 1

        with ROOT.TFile(cls.test_filename, "RECREATE") as f:
            tree = ROOT.TTree(cls.test_treename, cls.test_treename)

            x = array("f", [0])
            tree.Branch("myColumn", x, "myColumn/F")

            x[0] = 42
            tree.Fill()

            f.WriteObject(tree, cls.test_treename)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.test_filename)

    def test_getcolumnnames_from_strings(self):
        hn = create_dummy_headnode(self.test_treename, self.test_filename)
        proxy = Proxy.NodeProxy(hn)
        cn_vec = proxy.GetColumnNames()
        self.assertSequenceEqual(cn_vec, ["myColumn"])

    def test_getcolumnnames_from_rdatasetspec(self):
        spec = ROOT.RDF.Experimental.RDatasetSpec()
        spec.AddSample(("", self.test_treename, self.test_filename))

        hn = create_dummy_headnode(spec)
        proxy = Proxy.NodeProxy(hn)
        cn_vec = proxy.GetColumnNames()
        self.assertSequenceEqual(cn_vec, ["myColumn"])

    def test_getcolumnnames_after_define(self):
        """
        Check newly defined columns are available also locally.
        """

        node = create_dummy_headnode(1)
        proxy = Proxy.NodeProxy(node)

        cols_before = proxy.GetColumnNames()
        self.assertSequenceEqual(cols_before, [])

        proxy = proxy.Define("x", "42").Define("y", "43").Define("z", "44")

        cols_after = proxy.GetColumnNames()

        self.assertSequenceEqual(cols_after, ["x", "y", "z"])
        
    def test_get_column_type(self):
        
        """
        Check column type of the column in the dataset.
        """
        
        hn = create_dummy_headnode(self.test_treename, self.test_filename)
        proxy = Proxy.NodeProxy(hn)
        column_type = proxy.GetColumnType("myColumn")
        self.assertSequenceEqual(column_type, "Float_t")
    
    def test_get_column_type_after_define(self):
        """
        Check column type of the newly defined columns.
        """

        node = create_dummy_headnode(1)
        proxy = Proxy.NodeProxy(node)

        cols_before = proxy.GetColumnNames()
        self.assertSequenceEqual(cols_before, [])

        proxy = proxy.Define("x", "42.0").Define("y", "43")

        cols_after = proxy.GetColumnNames()
        column_types = []
        for column in cols_after:  
            column_type = proxy.GetColumnType(column)
            column_types.append(column_type)

        self.assertSequenceEqual(column_types, ["double", "int"])

    def test_columninfo_defines_twobranches(self):
        """
        Check new column names and types are available locally even if the same
        column name is used in different branches of the computation graph.
        """

        node = create_dummy_headnode(1)
        proxy = Proxy.NodeProxy(node)

        cols_before = proxy.GetColumnNames()
        self.assertSequenceEqual(cols_before, [])

        expected_coltype_1 = "Long64_t"
        branch_1 = proxy.Define("mycol", f"static_cast<{expected_coltype_1}>(42)")

        expected_coltype_2 = "float"
        branch_2 = proxy.Define("mycol", f"static_cast<{expected_coltype_2}>(33)")

        cols_1 = branch_1.GetColumnNames()
        self.assertSequenceEqual(cols_1, ["mycol"])
        coltype_1 = branch_1.GetColumnType(cols_1[0])
        self.assertEqual(coltype_1, expected_coltype_1)

        cols_2 = branch_2.GetColumnNames()
        self.assertSequenceEqual(cols_2, ["mycol"])
        coltype_2 = branch_2.GetColumnType(cols_2[0])
        self.assertEqual(coltype_2, expected_coltype_2)
