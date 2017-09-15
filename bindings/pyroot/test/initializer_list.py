import unittest
import ROOT

cppcode = """

"""

class InitializerList(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ROOT.gInterpreter.Declare(cppcode)
    def test_one(self):
        self.assertEqual(5 + 5, 10)

