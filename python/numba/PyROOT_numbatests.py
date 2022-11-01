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


if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    sys.exit(pytest.main(args=[__file__]))
