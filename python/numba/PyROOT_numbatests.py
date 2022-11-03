import sys
import os
import pytest
import time

import ROOT
import numba

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


@pytest.mark.skipif(
        not hasattr(numba, 'version_info') or numba.version_info < (0, 54),
        reason="Numba version 0.54 or more required")
class TestClasNumba:
    """Tests numba support for PyROOT"""

    def setup_class(cls):
        pass

    def compare(self, go_slow, go_fast, N, *args):
        t0 = time.time()
        for i in range(N):
            go_slow(*args)
        slow_time = time.time() - t0

        t0 = time.time()
        for i in range(N):
            go_fast(*args)
        fast_time = time.time() - t0

        return fast_time < slow_time

    def test01_simple_free_func(self):
        import ROOT.NumbaExt
        import math
        import numpy as np

        def go_slow(a):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += math.tanh(a[i, i])
            return a + trace

        @numba.jit(nopython=True)
        def go_fast(a):
            trace = 0.0
            for i in range(a.shape[0]):
                trace += ROOT.tanh(a[i, i])
            return a + trace

        x = np.arange(100, dtype=np.float64).reshape(10, 10)

        assert (go_fast(x) == go_slow(x)).all()
        assert self.compare(go_slow, go_fast, 300000, x)

    def test02_member_function(self):
        import ROOT.NumbaExt
        import math

        # Obtain a vector of ROOT::Math::LorentzVector from the sample
        # .root file
        myfile = ROOT.TFile.Open("vec_lv.root")
        vec_lv = myfile.vecOfLV

        def calc_pt(lv):
            return math.sqrt(lv.Px() ** 2 + lv.Py() ** 2)

        def calc_pt_vec(vec_lv):
            pt = []
            for i in range(vec_lv.size()):
                pt.append((calc_pt(vec_lv[i]), vec_lv[i].Pt()))
            return pt

        @numba.njit
        def numba_calc_pt(lv):
            return math.sqrt(lv.Px() ** 2 + lv.Py() ** 2)

        def numba_calc_pt_vec(vec_lv):
            pt = []
            for i in range(vec_lv.size()):
                pt.append((numba_calc_pt(vec_lv[i]), vec_lv[i].Pt()))
            return pt

        assert (False not in
                tuple(x == y for x, y in numba_calc_pt_vec(vec_lv)))
        assert self.compare(calc_pt_vec, numba_calc_pt_vec, 1, vec_lv)


if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    sys.exit(pytest.main(args=[__file__]))
