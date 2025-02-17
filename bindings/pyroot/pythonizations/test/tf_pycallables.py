"""
Tests for passing Python callables when constructing TFX classes.

This feature is not implemented by a PyROOT pythonization, but by a converter of
Cppyy that creates a C++ wrapper to invoke the Python callable.
"""

import unittest
import math

import ROOT


def pyf_tf1_identity(x, p):
    return x[0]


def pyf_tf1_params(x, p):
    return p[0] * x[0] + p[1]


class pyf_tf1_callable:
    def __call__(self, x, p):
        return p[0] * x[0] + p[1]

def pyf_func(x, pars):
    return pars[0] * x[0] * x[2] + x[1] * pars[1]

def pyf_tf1_gauss(x, p):
    return p[0] * 1.0 / math.sqrt(2.0 * math.pi * p[2]**2) * math.exp(-(x[0] - p[1])**2 / 2.0 / p[2]**2)

def pyf_tf1_coulomb(x, p):
    return p[1] * x[0] * x[1] / (p[0]**2) * math.exp(-p[2] / p[0])


class TF1(unittest.TestCase):
    """
    Test passing Python callables to ROOT::TF1
    """

    def test_identity(self):
        """
        Test simple function without parameters
        """
        f = ROOT.TF1("tf1_identity", pyf_tf1_identity, 0.0, 1.0)
        for x in [0.0, -1.0, 42.0]:
            self.assertEqual(f.Eval(x), x)

    def test_params(self):
        """
        Test function with parameters
        """
        npars = 2
        f = ROOT.TF1("tf1_params", pyf_tf1_params, 0.0, 1.0, npars)
        par1 = 2.0
        par2 = -1.0
        f.SetParameter(0, par1)
        f.SetParameter(1, par2)
        for x in [0.0, -1.0, 42.0]:
            self.assertEqual(f.Eval(x), pyf_tf1_params([x], [par1, par2]))

    def test_callable(self):
        """
        Test function provided as callable
        """
        npars = 2
        pycallable = pyf_tf1_callable()
        f = ROOT.TF1("tf1_callable", pycallable, 0.0, 1.0, npars)
        par1 = 2.0
        par2 = -1.0
        f.SetParameter(0, par1)
        f.SetParameter(1, par2)
        for x in [0.0, -1.0, 42.0]:
            self.assertEqual(f.Eval(x), pycallable([x], [par1, par2]))


    def test_fitgauss(self):
        """
        Test fitting a histogram to a Python function
        """
        # Gaus function
        f = ROOT.TF1("tf1_fitgauss", pyf_tf1_gauss, -4, 4, 3)
        f.SetParameter(0, 10.0) # scale
        f.SetParameter(1, -1.0) # mean
        f.SetParameter(2, 2.0) # standard deviation

        # Sample gauss in histogram
        h = ROOT.TH1F("h", "test", 100, -4, 4)
        h.FillRandom("gaus", 100000)
        h.Scale(1.0 / 100000.0 * 100.0 / 8.0) # Normalize as density

        # Fit to histogram and get parameters
        h.Fit( f, "0Q" )
        scale = f.GetParameter(0)
        mean = f.GetParameter(1)
        std = f.GetParameter(2)

        self.assertAlmostEqual(scale, 1.0, 2)
        self.assertAlmostEqual(mean, 0.0, 2)
        self.assertAlmostEqual(abs(std), 1.0, 2)

    def test_evalpar(self):
        """
        Test the 2D Numpy array pythonizations for TF1::EvalPar
        """
        import numpy as np

        rtf1_coulomb = ROOT.TF1("my_func", pyf_tf1_coulomb, -10, 10)

        # x dataset: 5 pairs of particle charges
        x = np.array([
            [1.0, 10, 2.0],
            [1.5, 10, 2.5],
            [2.0, 10, 3.0],
            [2.5, 10, 3.5],
            [3.0, 10, 4.0]
        ])

        params = np.array([
            [1.0],       # Distance between charges r
            [8.99e9],    # Coulomb constant k (in N·m²/C²)
            [0.1]        # Additional factor for modulation
        ])

        # Slice to avoid the dummy column of 10's
        res = rtf1_coulomb.EvalPar(x[:, ::2], params)

        for i in range(len(x)):
            expected_value = pyf_tf1_coulomb(x[i, ::2], params)
            self.assertEqual(res[i], expected_value)
    
    def test_evalpar_dynamic(self):
        """
        Test the 2D NumPy pythonizations with dynamic TF1 data dimensions
        """
        import numpy as np

        # Here we do not set the ndims, defaults to 1
        rtf1_func = ROOT.TF1("my_func", pyf_func, -10, 10)

        # x dataset with ndims 3
        x = np.array([[2., 2, 1],
                [1., 2, 3],
                [2., 2, 1],
                [4., 3, 2]])

        pars = np.array([2., 3.])
        res = rtf1_func.EvalPar(x, pars)

        for i in range(len(x)):
            expected_value = pyf_func(x[i], pars)
            self.assertEqual(res[i], expected_value)


def pyf_tf2_params(x, p):
    return p[0] * x[0] + p[1] * x[1] + p[2]


class TF2(unittest.TestCase):
    """
    Test passing Python callables to ROOT::TF2
    """

    def test_params(self):
        """
        Test function with parameters
        """
        npars = 3
        f = ROOT.TF1("tf2_params", pyf_tf2_params, 0.0, 1.0, npars)
        par1 = 2.0
        par2 = -1.0
        par3 = 1.0
        f.SetParameter(0, par1)
        f.SetParameter(1, par2)
        f.SetParameter(2, par3)
        for x in [(0.0, 0.0), (-1.0, 1.0), (42.0, 0.0)]:
            self.assertEqual(f.Eval(*x), pyf_tf2_params(x, [par1, par2, par3]))


def pyf_tf3_params(x, p):
    return p[0] * x[0] + p[1] * x[1] + p[2] * x[2] + p[3]


class TF3(unittest.TestCase):
    """
    Test passing Python callables to ROOT::TF3
    """

    def test_params(self):
        """
        Test function with parameters
        """
        npars = 4
        f = ROOT.TF1("tf2_params", pyf_tf2_params, 0.0, 1.0, npars)
        par1 = 2.0
        par2 = -1.0
        par3 = 1.0
        par4 = 3.0
        f.SetParameter(0, par1)
        f.SetParameter(1, par2)
        f.SetParameter(2, par3)
        f.SetParameter(3, par4)
        for x in [(0.0, 0.0, 0.0), (-1.0, 1.0, 2.0), (42.0, 0.0, -10.0)]:
            self.assertEqual(f.Eval(*x), pyf_tf2_params(x, [par1, par2, par3, par4]))


if __name__ == '__main__':
    unittest.main()

