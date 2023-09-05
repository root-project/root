import unittest
import array

import ROOT


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


if __name__ == '__main__':
    unittest.main()
