import unittest

import ROOT
import numpy as np

class TRandomNumpyArrays(unittest.TestCase):
    """
    Test for pythonizations thst allow filling numpy arrays with TRandom scalar
    RNGs.
    We only cover the technical aspect, ensuring the array if filled. With the seed
    set, all executions should be repeatable.
    """

    # Tests
    def test_binomial(self):
        r = ROOT.TRandom2(123)
        vi = np.zeros(100, dtype=np.int32)
        r.BinomialN(100, vi, 123, 0.12345)
        self.assertEqual(np.count_nonzero(vi), len(vi))

    def test_breit_wigner(self):
        r = ROOT.TRandom2(123)
        vd = np.zeros(100, dtype=np.float64)
        r.BreitWignerN(100, vd, 123, 0.12345)
        self.assertEqual(np.count_nonzero(vd), len(vd))

    def test_exp(self):
        r = ROOT.TRandom2(123)
        vd = np.zeros(100, dtype=np.float64)
        r.ExpN(100, vd, 1.5)
        self.assertEqual(np.count_nonzero(vd), len(vd))

    def test_gaus(self):
        r = ROOT.TRandom2(123)
        vd = np.zeros(100, dtype=np.float64)
        r.GausN(100, vd, 5, 1.5)
        self.assertEqual(np.count_nonzero(vd), len(vd))

    def test_integer(self):
        r = ROOT.TRandom2(123)
        vui = np.zeros(100, dtype=np.uint32)
        r.IntegerN(100, vui, 15)
        self.assertGreater(np.count_nonzero(vui), 0.9 * len(vui))

    def test_landau(self):
        r = ROOT.TRandom2(123)
        vd = np.zeros(100, dtype=np.float64)
        r.LandauN(100, vd, 5, 1.5)
        self.assertEqual(np.count_nonzero(vd), len(vd))

    def test_poisson(self):
        r = ROOT.TRandom2(123)
        vull = np.zeros(100, dtype=np.ulonglong)
        r.PoissonN(100, vull, 15)
        self.assertGreater(np.count_nonzero(vull), 0.9 * len(vull))

    def test_poissond_d(self):
        r = ROOT.TRandom2(123)
        vd = np.zeros(100, dtype=np.float64)
        r.PoissonDN(100, vd, 15)
        self.assertGreater(np.count_nonzero(vd), 0.9 * len(vd))

    def test_uniform(self):
        r = ROOT.TRandom2(123)
        vd = np.zeros(100, dtype=np.float64)
        r.UniformN(100, vd, 5.0)
        self.assertEqual(np.count_nonzero(vd), len(vd))

    def test_uniform_2_params(self):
        r = ROOT.TRandom2(123)
        vd = np.zeros(100, dtype=np.float64)
        r.UniformN(100, vd, 5, 1.5)
        self.assertEqual(np.count_nonzero(vd), len(vd))

    def test_pythonization_arrays(self):
        r = ROOT.TRandom(123)
        aa=numpy.zeros([2,3,4])
        r.Gaus(4, 0.4, out=aa)
        r.Gaus(4, 0.4)
        r.Gaus(4, 0.4, size=[2,3,4])

    def test_pythonization_all_wrappers(self):
        r = ROOT.TRandom3(123)
        r.Binomial(size=[2,3], ntot=25, prob=0.123)
        r.BreitWigner(size=[2,3], mean=3, gamma=0.789)
        r.Exp(size=[2,3], tau=1000)
        r.Integer(size=[2,3], imax=10)
        r.Landau(size=[2,3], mean=5, sigma=2)
        r.Poisson(size=[2,3], mean=5)
        r.PoissonD(size=[2,3], mean=5)
        r.Uniform(size=[2,3], x2=5)
        r.Uniform(size=[2,3], x2=-10, x1=5)