import unittest
import ROOT

cppcode = """
struct A {
  A(int _i, double _f) : i(_i), f(_f) {}
  A(int _i) : i(_i), f(99.99) {}
  int i;
  double f;
};
int get_i(const A& a) { return a.i; }
double get_f(const A& a) { return a.f; }
"""

class ListInitialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ROOT.gInterpreter.Declare(cppcode)

    def test_function_call(self):
        self.assertEqual(ROOT.get_i((10,20.)), 10)
        self.assertEqual(ROOT.get_f((10,20.)), 20.)
        self.assertAlmostEqual(ROOT.get_f((10,)), 99.99)
    
    def test_invalid_constructor(self):
        with self.assertRaises(TypeError): ROOT.get_i((1,2,3))
        with self.assertRaises(TypeError): ROOT.get_i((1,'abc'))

if __name__ == '__main__':
    unittest.main()