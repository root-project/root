import array
import os
import unittest

import ROOT
from ROOT import TClass, TComplex, TDirectory, TObject, TObjString


class TClassDynamicCast(unittest.TestCase):
    """
    Test for the pythonization of TClass::DynamicCast, which adds an
    an extra cast before returning the Python proxy to the user so that
    it has the right type.
    """

    # Tests
    def test_dynamiccast(self):
        tobj_class = TClass.GetClass("TObject")
        tobjstr_class = TClass.GetClass("TObjString")

        o = TObjString("a")

        # Upcast: TObject <- TObjString
        o_upcast = tobjstr_class.DynamicCast(tobj_class, o)
        self.assertEqual(type(o_upcast), TObject)

        # Downcast: TObject -> TObjString
        o_downcast = tobjstr_class.DynamicCast(tobj_class, o_upcast, False)
        self.assertEqual(type(o_downcast), TObjString)

    def test_dynamiccast_offset(self):
        # The second base of a class with multiple inheritance is at a non-zero
        # offset, which the cast has to apply in both directions.
        ROOT.gInterpreter.Declare("""
        struct DynamicCastBase1 { virtual ~DynamicCastBase1() {} int fBase1 = 1; };
        struct DynamicCastBase2 { virtual ~DynamicCastBase2() {} int fBase2 = 2; };
        struct DynamicCastDerived : DynamicCastBase1, DynamicCastBase2 {};
        """)
        base_class = TClass.GetClass("DynamicCastBase2")
        derived_class = TClass.GetClass("DynamicCastDerived")

        o = ROOT.DynamicCastDerived()

        # Upcast: DynamicCastBase2 <- DynamicCastDerived
        o_upcast = derived_class.DynamicCast(base_class, o)
        self.assertEqual(type(o_upcast), ROOT.DynamicCastBase2)
        self.assertNotEqual(ROOT.addressof(o_upcast), ROOT.addressof(o))
        self.assertEqual(o_upcast.fBase2, 2)

        # Downcast: DynamicCastBase2 -> DynamicCastDerived
        o_downcast = derived_class.DynamicCast(base_class, o_upcast, False)
        self.assertEqual(type(o_downcast), ROOT.DynamicCastDerived)
        self.assertEqual(ROOT.addressof(o_downcast), ROOT.addressof(o))

        # The object can also be given by its address
        o_upcast = derived_class.DynamicCast(base_class, ROOT.addressof(o))
        self.assertEqual(o_upcast.fBase2, 2)


class TContextContextManager(unittest.TestCase):
    """
    Test of TContext used as context manager
    """

    def default_constructor(self):
        """
        Check status of gDirectory with default constructor.
        """
        filename = "TContextContextManager_test_default_constructor.root"
        self.assertEqual(ROOT.gDirectory, ROOT.gROOT)

        with TDirectory.TContext():
            # Create a file to change gDirectory
            testfile = ROOT.TFile(filename, "recreate")
            self.assertEqual(ROOT.gDirectory, testfile)
            testfile.Close()

        self.assertEqual(ROOT.gDirectory, ROOT.gROOT)
        os.remove(filename)

    def constructor_onearg(self):
        """
        Check status of gDirectory with constructor taking a new directory.
        """
        filenames = ["TContextContextManager_test_constructor_onearg_{}.root".format(i) for i in range(2)]

        file0 = ROOT.TFile(filenames[0], "recreate")
        file1 = ROOT.TFile(filenames[1], "recreate")
        self.assertEqual(ROOT.gDirectory, file1)

        with TDirectory.TContext(file0):
            self.assertEqual(ROOT.gDirectory, file0)

        self.assertEqual(ROOT.gDirectory, file1)
        file0.Close()
        file1.Close()
        for filename in filenames:
            os.remove(filename)

    def constructor_twoargs(self):
        """
        Check status of gDirectory with constructor taking the previous directory and a new one.
        """
        filenames = ["TContextContextManager_test_constructor_onearg_{}.root".format(i) for i in range(3)]

        file0 = ROOT.TFile(filenames[0], "recreate")
        file1 = ROOT.TFile(filenames[1], "recreate")
        file2 = ROOT.TFile(filenames[2], "recreate")
        self.assertEqual(ROOT.gDirectory, file2)

        with TDirectory.TContext(file0, file1):
            self.assertEqual(ROOT.gDirectory, file1)

        self.assertEqual(ROOT.gDirectory, file0)
        file0.Close()
        file1.Close()
        file2.Close()
        for filename in filenames:
            os.remove(filename)

    def test_all(self):
        """
        Run all tests of this class sequentially.
        The tests of this class rely on the current directory, which can be changed
        unpredictably if they are run concurrently.
        """
        self.default_constructor()
        self.constructor_onearg()
        self.constructor_twoargs()


class TestTComplexOperators(unittest.TestCase):
    """
    Test for the operators of TComplex:
    __radd__, __rsub__, __rmul__, __rtruediv__/__rdiv__.
    """

    c = TComplex(4.,0)

    d = 2.

    s = 'string'

    # check the expected result for d + c and that Re(c + d) == Re(d + c)
    def test_radd(self):
        self.assertEqual((self.d + self.c).Re(), 6.0)
        self.assertEqual((self.c + self.d).Re(), (self.d + self.c).Re())

    # check the expected result for d - c and that Re(c - d) == -Re(d - c)
    def test_rsub(self):
        self.assertEqual((self.d - self.c).Re(), -2.0)
        self.assertEqual((self.c - self.d).Re(), -((self.d - self.c).Re()))

    # check the expected result for d * c and that Re(c * d) == Re(d * c)
    def test_rmul(self):
        self.assertEqual((self.d * self.c).Re(), 8.0)
        self.assertEqual((self.c * self.d).Re(), (self.d * self.c).Re())

    # check the expected result for d / c
    def test_rdiv(self):
        self.assertEqual((self.d / self.c).Re(), 0.5)
        with self.assertRaises(TypeError):
            self.s / self.c


class TIterIterator(unittest.TestCase):
    """
    Test for the pythonization that allows instances of TIter to
    behave as Python iterators.
    """

    num_elems = 3

    # Helpers
    def create_tcollection(self):
        c = ROOT.TList()
        for _ in range(self.num_elems):
            o = ROOT.TObject()
            # Prevent immediate deletion of C++ TObjects
            ROOT.SetOwnership(o, False)
            c.Add(o)

        return c

    # Tests
    def test_iterable(self):
        # Check that TIter instances are iterable
        c = self.create_tcollection()

        itc = ROOT.TIter(c)
        # An iterator of an iterator is itself
        self.assertEqual(itc, iter(itc))

    def test_iterator(self):
        # Check that TIter instances are iterators
        c = self.create_tcollection()

        itc1 = ROOT.TIter(c)
        itc2 = ROOT.TIter(c)
        for _ in range(c.GetEntries()):
            self.assertIs(next(itc1), itc2.Next())

    def test_for_loop_syntax(self):
        # Somehow redundant, but good to test with real syntax
        c = self.create_tcollection()

        itc1 = ROOT.TIter(c)
        itc2 = ROOT.TIter(c)
        for elem1, elem2 in zip(itc1, itc2):
            self.assertIs(elem1, elem2)


class TGraphGetters(unittest.TestCase):
    """
    Test for the pythonization of TGraph, TGraph2D and their error
    subclasses, in particular of their X,Y,Z coordinates and errors
    getters, which sets the size of the returned buffers.
    """

    # Tests
    def test_graph(self):
        N = 5
        xval, yval = 1, 2

        ax = array.array('d', map(lambda x: x*xval, range(N)))
        ay = array.array('d', map(lambda x: x*yval, range(N)))

        g = ROOT.TGraph(N, ax, ay)

        # x and y are buffers of doubles
        x = g.GetX()
        y = g.GetY()

        # We can get the size of the buffers
        self.assertEqual(len(x), N)
        self.assertEqual(len(y), N)

        # The buffers are iterable
        self.assertEqual(list(x), list(ax))
        self.assertEqual(list(y), list(ay))

    def test_graph2derrors(self):
        N = 5
        xval, yval, zval = 1, 2, 3
        xerrval, yerrval, zerrval = 0.1, 0.2, 0.3

        ax = array.array('d', map(lambda x: x*xval, range(N)))
        ay = array.array('d', map(lambda x: x*yval, range(N)))
        az = array.array('d', map(lambda x: x*zval, range(N)))
        aex = array.array('d', map(lambda x: x*xerrval, range(N)))
        aey = array.array('d', map(lambda x: x*yerrval, range(N)))
        aez = array.array('d', map(lambda x: x*zerrval, range(N)))

        g = ROOT.TGraph2DErrors(N, ax, ay, az, aex, aey, aez)

        # x, y, z, ex, ey and ez are buffers of doubles
        x = g.GetX()
        y = g.GetY()
        z = g.GetZ()
        ex = g.GetEX()
        ey = g.GetEY()
        ez = g.GetEZ()

        # We can get the size of the buffers
        self.assertEqual(len(x), N)
        self.assertEqual(len(y), N)
        self.assertEqual(len(z), N)
        self.assertEqual(len(ex), N)
        self.assertEqual(len(ey), N)
        self.assertEqual(len(ez), N)

        # The buffers are iterable
        self.assertEqual(list(x), list(ax))
        self.assertEqual(list(y), list(ay))
        self.assertEqual(list(z), list(az))
        self.assertEqual(list(ex), list(aex))
        self.assertEqual(list(ey), list(aey))
        self.assertEqual(list(ez), list(aez))

    def test_graphasymmerrors(self):
        n = 10
        ax = array.array('d', [0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95])
        ay = array.array('d', [1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1])
        aexl = array.array('d', [.05, .1, .07, .07, .04, .05, .06, .07, .08, .05])
        aeyl = array.array('d', [.8, .7, .6, .5, .4, .4, .5, .6, .7, .8])
        aexh = array.array('d', [.02, .08, .05, .05, .03, .03, .04, .05, .06, .03])
        aeyh = array.array('d', [.6, .5, .4, .3, .2, .2, .3, .4, .5, .6])
        g = ROOT.TGraphAsymmErrors(n, ax, ay, aexl, aexh, aeyl, aeyh)

        # All of the next calls return C-style arrays of doubles
        # In cppyy they are converted to 'LowLevelView' objects
        # The Pythonizations of the methods make sure to call 'reshape'
        # So that cppyy can understand the shape of the arrays.
        x = g.GetX()
        y = g.GetY()
        exlow = g.GetEXlow()
        eylow = g.GetEYlow()
        exhigh = g.GetEXhigh()
        eyhigh = g.GetEYhigh()

        self.assertEqual(len(x), n)
        self.assertEqual(len(y), n)
        self.assertEqual(len(exlow), n)
        self.assertEqual(len(eylow), n)
        self.assertEqual(len(exhigh), n)
        self.assertEqual(len(eyhigh), n)

        self.assertEqual(list(x), list(ax))
        self.assertEqual(list(y), list(ay))
        self.assertEqual(list(exlow), list(aexl))
        self.assertEqual(list(eylow), list(aeyl))
        self.assertEqual(list(exhigh), list(aexh))
        self.assertEqual(list(eyhigh), list(aeyh))

    def test_graphbenterrors(self):
        n = 10
        ax = array.array('d', [0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95])
        ay = array.array('d', [1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1])
        aexl = array.array('d', [.05, .1, .07, .07, .04, .05, .06, .07, .08, .05])
        aeyl = array.array('d', [.8, .7, .6, .5, .4, .4, .5, .6, .7, .8])
        aexh = array.array('d', [.02, .08, .05, .05, .03, .03, .04, .05, .06, .03])
        aeyh = array.array('d', [.6, .5, .4, .3, .2, .2, .3, .4, .5, .6])
        aexld = array.array('d', [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0])
        aeyld = array.array('d', [.0, .0, .05, .0, .0, .0, .0, .0, .0, .0])
        aexhd = array.array('d', [.0, .0, .0, .0, .0, .0, .0, .0, .0, .0])
        aeyhd = array.array('d', [.0, .0, .0, .0, .0, .0, .0, .0, .05, .0])
        g = ROOT.TGraphBentErrors(n, ax, ay, aexl, aexh, aeyl, aeyh, aexld, aexhd, aeyld, aeyhd)

        # All of the next calls return C-style arrays of doubles
        # In cppyy they are converted to 'LowLevelView' objects
        # The Pythonizations of the methods make sure to call 'reshape'
        # So that cppyy can understand the shape of the arrays.
        x = g.GetX()
        y = g.GetY()
        exlow = g.GetEXlow()
        eylow = g.GetEYlow()
        exhigh = g.GetEXhigh()
        eyhigh = g.GetEYhigh()
        exlowd = g.GetEXlowd()
        exhighd = g.GetEXhighd()
        eylowd = g.GetEYlowd()
        eyhighd = g.GetEYhighd()

        self.assertEqual(len(x), n)
        self.assertEqual(len(y), n)
        self.assertEqual(len(exlow), n)
        self.assertEqual(len(eylow), n)
        self.assertEqual(len(exhigh), n)
        self.assertEqual(len(eyhigh), n)
        self.assertEqual(len(exlowd), n)
        self.assertEqual(len(exhighd), n)
        self.assertEqual(len(eylowd), n)
        self.assertEqual(len(eyhighd), n)

        self.assertEqual(list(x), list(ax))
        self.assertEqual(list(y), list(ay))
        self.assertEqual(list(exlow), list(aexl))
        self.assertEqual(list(eylow), list(aeyl))
        self.assertEqual(list(exhigh), list(aexh))
        self.assertEqual(list(eyhigh), list(aeyh))

        self.assertEqual(list(exlowd), list(aexld))
        self.assertEqual(list(exhighd), list(aexhd))
        self.assertEqual(list(eylowd), list(aeyld))
        self.assertEqual(list(eyhighd), list(aeyhd))

    def test_graphmultierrors(self):
        n = 10
        ax = array.array('d', [0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95])
        ay = array.array('d', [1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1])
        aexl = array.array('d', [.05, .1, .07, .07, .04, .05, .06, .07, .08, .05])
        aeyl = array.array('d', [.8, .7, .6, .5, .4, .4, .5, .6, .7, .8])
        aexh = array.array('d', [.02, .08, .05, .05, .03, .03, .04, .05, .06, .03])
        aeyh = array.array('d', [.6, .5, .4, .3, .2, .2, .3, .4, .5, .6])
        g = ROOT.TGraphMultiErrors("gme", "TGraphMultiErrors Example", n, ax, ay, aexl, aexh, aeyl, aeyh)

        # All of the next calls return C-style arrays of doubles
        # In cppyy they are converted to 'LowLevelView' objects
        # The Pythonizations of the methods make sure to call 'reshape'
        # So that cppyy can understand the shape of the arrays.
        x = g.GetX()
        y = g.GetY()
        exlow = g.GetEXlow()
        eylow = g.GetEYlow()
        exhigh = g.GetEXhigh()
        eyhigh = g.GetEYhigh()

        self.assertEqual(len(x), n)
        self.assertEqual(len(y), n)
        self.assertEqual(len(exlow), n)
        self.assertEqual(len(eylow), n)
        self.assertEqual(len(exhigh), n)
        self.assertEqual(len(eyhigh), n)

        self.assertEqual(list(x), list(ax))
        self.assertEqual(list(y), list(ay))
        self.assertEqual(list(exlow), list(aexl))
        self.assertEqual(list(eylow), list(aeyl))
        self.assertEqual(list(exhigh), list(aexh))
        self.assertEqual(list(eyhigh), list(aeyh))


class TParameterTemplateInstantiations(unittest.TestCase):
    """
    Instantiating TParameter<T> for a type without a dictionary (e.g. char)
    from the interpreter used to fail with unresolved
    TParameter<T>::Class() and TParameter<T>::Streamer(TBuffer&) symbols,
    see https://github.com/root-project/root/issues/10724. Check that all
    standard arithmetic types can be constructed, printed and serialized
    from the interpreter.
    """

    # (type name, value passed to the constructor, value expected from GetVal())
    # cppyy maps the character types to Python str, so 42 comes back as "*"
    instantiations = [
        ("bool", True, True),
        ("char", 42, "*"),
        ("signed char", 42, "*"),
        ("unsigned char", 42, "*"),
        ("short", 42, 42),
        ("unsigned short", 42, 42),
        ("int", 42, 42),
        ("unsigned int", 42, 42),
        ("long", 42, 42),
        ("unsigned long", 42, 42),
        ("long long", 42, 42),
        ("unsigned long long", 42, 42),
        ("float", 0.5, 0.5),
        ("double", 0.5, 0.5),
    ]

    def test_construct_and_print(self):
        for typename, val, expected in self.instantiations:
            with self.subTest(typename=typename):
                param = ROOT.TParameter[typename]("p", val)
                self.assertEqual(param.GetVal(), expected)
                # Print and ls write to std::cout; exercise the code path
                # (they used to be uncallable without arguments)
                param.Print()
                param.ls()

    def test_has_dictionary(self):
        for typename, _, _ in self.instantiations:
            with self.subTest(typename=typename):
                tclass = ROOT.TClass.GetClass(f"TParameter<{typename}>")
                self.assertIsNotNone(tclass)
                self.assertTrue(tclass.HasDictionary())

    def test_io_roundtrip(self):
        for typename, val, expected in self.instantiations:
            with self.subTest(typename=typename):
                param = ROOT.TParameter[typename]("p", val)
                memfile = ROOT.TMemFile("TParameterTemplateInstantiations.root", "recreate")
                memfile.WriteObject(param, "p")
                param_read = memfile.Get("p")
                self.assertEqual(param_read.GetVal(), expected)
                memfile.Close()


class Float16Double32Conversions(unittest.TestCase):
    """
    Float16_t and Double32_t are typedefs to float and double that only
    differ in how ROOT I/O stores them. They have to convert to and from
    Python like the types they alias, by value and by reference.
    """

    @classmethod
    def setUpClass(cls):
        ROOT.gInterpreter.Declare("""
        struct Float16Double32Holder {
           Float16_t fF = 0.5f;
           Double32_t fD = 1.5;
           Float16_t GetF() { return fF; }
           Double32_t GetD() { return fD; }
           Float16_t &RefF() { return fF; }
           Double32_t &RefD() { return fD; }
           float TakeF(Float16_t x) { return x; }
           double TakeD(Double32_t x) { return x; }
           float TakeConstRefF(const Float16_t &x) { return x; }
           double TakeConstRefD(const Double32_t &x) { return x; }
        };
        """)

    def test_conversions(self):
        h = ROOT.Float16Double32Holder()
        self.assertAlmostEqual(h.fF, 0.5)
        self.assertAlmostEqual(h.fD, 1.5)
        self.assertAlmostEqual(h.GetF(), 0.5)
        self.assertAlmostEqual(h.GetD(), 1.5)
        self.assertAlmostEqual(h.RefF(), 0.5)
        self.assertAlmostEqual(h.RefD(), 1.5)
        self.assertAlmostEqual(h.TakeF(2.5), 2.5)
        self.assertAlmostEqual(h.TakeD(3.5), 3.5)
        self.assertAlmostEqual(h.TakeConstRefF(4.5), 4.5)
        self.assertAlmostEqual(h.TakeConstRefD(5.5), 5.5)

    def test_vector(self):
        # Spell the template argument both as a string and as the Python
        # proxy of the typedef, which may resolve through different code paths
        for tp in ["Float16_t", "Double32_t", ROOT.Float16_t, ROOT.Double32_t]:
            with self.subTest(tp=tp):
                v = ROOT.std.vector[tp]([0.5, 1.5])
                self.assertEqual(len(v), 2)
                self.assertAlmostEqual(v[0], 0.5)
                self.assertAlmostEqual(v[1], 1.5)


if __name__ == '__main__':
    unittest.main()
