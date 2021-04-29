import unittest

import DistRDF
import ROOT
from DistRDF.Backends import Base
from DistRDF.Backends import Utils


class BackendInitTest(unittest.TestCase):
    """Backend abstract class cannot be instantiated."""

    def test_backend_init_error(self):
        """
        Any attempt to instantiate the `Backend` abstract class results in
        a `TypeError`.

        """
        with self.assertRaises(TypeError):
            Base.BaseBackend()

    def test_subclass_without_method_error(self):
        """
        Creation of a subclass without implementing `execute` method throws
        a `TypeError`.

        """
        class TestBackend(Base.BaseBackend):
            pass

        with self.assertRaises(TypeError):
            TestBackend()


class DeclareHeadersTest(unittest.TestCase):
    """Static method 'declare_headers' in Backend class."""

    def test_single_header_declare(self):
        """'declare_headers' with a single header to be included."""
        Utils.declare_headers(["test_headers/header1.hxx"])

        self.assertEqual(ROOT.f(1), True)

    def test_multiple_headers_declare(self):
        """'declare_headers' with multiple headers to be included."""
        Utils.declare_headers(["test_headers/header2.hxx",
                               "test_headers/header3.hxx"])

        self.assertEqual(ROOT.a(1), True)
        self.assertEqual(ROOT.f1(2), 2)
        self.assertEqual(ROOT.f2("myString"), "myString")

    def test_header_declaration_on_current_session(self):
        """Header has to be declared on the current session"""
        # Before the header declaration the function f is not present on the
        # ROOT interpreter
        with self.assertRaises(AttributeError):
            self.assertRaises(ROOT.b(1))
        Utils.declare_headers(["test_headers/header4.hxx"])
        self.assertEqual(ROOT.b(1), True)


class InitializationTest(unittest.TestCase):
    """Check the initialize method"""

    def test_initialization_runs_in_current_environment(self):
        """
        User initialization method should be executed on the current user
        session, so actions applied by the user initialization function are
        also visible in the current scenario.
        """
        def defineIntVariable(name, value):
            import ROOT
            ROOT.gInterpreter.ProcessLine("int %s = %s;" % (name, value))

        varvalue = 2
        DistRDF.initialize(defineIntVariable, "myInt", varvalue)
        self.assertEqual(ROOT.myInt, varvalue)
