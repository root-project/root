# Author: Aaron Jomy CERN 09/2024
# Author: Vincenzo Eduardo Padulano CERN 09/2024

################################################################################
# Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
/**
\class TRandom
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
## PyROOT

The TRandom class has several additions for its use from Python, which are also
available in its subclasses TRandom2 and TRandom3. The python interface is modeled
on the interface of numpy.random.Generator.random.

This is currently limited to the distributions or 1-dimensional values. 

Scalar or arrays can be generated:
\code{.py}
>>> import ROOT as RT
>>> import numpy
>>> r = RT.TRandom(123)
>>> aa=numpy.zeros([2,3,4])
>>> r.Gaus(4, 0.4, out=aa)
>>> aa
array([[[3.96642121, 4.08632025, 4.26221681, 3.96809844],
        [3.67775501, 4.22637515, 3.46672227, 4.34481581],
        [4.12859111, 3.57025717, 3.84007792, 3.82163463]],

       [[4.26388937, 4.01280784, 4.35675554, 3.52723725],
        [3.32696688, 4.19441549, 4.28081704, 3.72076935],
        [3.59817384, 3.42698926, 4.69339572, 3.97866244]]])
>>> r.Gaus(4, 0.4)
5.311091735110202
>>> r.Gaus(4, 0.4, size=[2,3,4])
array([[[3.89629527, 4.01775263, 3.51479162, 4.05398529],
        [4.12382313, 3.65945926, 4.02825931, 4.85850178],
        [3.99641824, 3.70762586, 3.9692221 , 4.15951614]],

       [[3.66456567, 3.68396422, 3.7158251 , 3.90612771],
        [4.17368496, 3.63361176, 3.92615234, 3.99501832],
        [3.72500961, 3.75392728, 4.00470541, 3.96299244]]])
\endcode

The covered generators are:
\code{.py}
>>> r.Binomial(size=[2,3], ntot=25, prob=0.123)
array([[4, 2, 5],
       [1, 3, 2]], dtype=int32)
>>> r.BreitWigner(size=[2,3], mean=3, gamma=0.789)
array([[2.10769386, 3.37180071, 2.75851619],
       [3.08555421, 2.77677075, 2.7986828 ]])
>>> r.Exp(size=[2,3], tau=1000)
array([[ 255.33683037, 2115.41171346,  243.69722096],
       [1552.28132836,  350.13630267,  509.41190404]])
>>> r.Integer(size=[2,3], imax=10)
array([[7, 4, 3],
       [6, 1, 1]], dtype=uint32)
>>> r.Landau(size=[2,3], mean=5, sigma=2)
array([[ 8.61497464,  5.11830138,  3.93617892],
       [ 2.08862153,  4.43892651, 13.53280511]])
>>> r.Poisson(size=[2,3], mean=5)
array([[ 5,  7,  5],
       [ 7,  2, 10]], dtype=uint64)
>>> r.PoissonD(size=[2,3], mean=5)
array([[2., 7., 6.],
       [5., 7., 4.]])
>>> r.Uniform(size=[2,3], x2=5)
array([[2.1122357 , 0.22695246, 1.04537933],
       [1.36725521, 4.89698287, 0.21730294]])
>>> r.Uniform(size=[2,3], x2=-10, x1=5)
array([[-0.73856891, -5.70391524, -9.68524487],
       [-5.63854465,  1.3690705 , -0.07928077]])
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

# Pythonic wrappers to TRandom based on the interface of numpy.random.Generator.random()
# random.Generator.random(size=None, dtype=np.float64, out=None)

# out / size mismatch leads to:
# >>> g.random(out=out, size=s2)
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "numpy/random/_generator.pyx", line 356, in numpy.random._generator.Generator.random
#   File "_common.pyx", line 304, in numpy.random._common.double_fill
#   File "_common.pyx", line 288, in numpy.random._common.check_output
# ValueError: size must match out.shape when used together

from . import pythonization

def _TRandom_Binomial(self, ntot, prob, size=None, out=None):

    import ROOT
    import numpy

    if size == None and out == None:
        return self._Binomial(ntot, prob)
    if out != None:
        if out.dtype != numpy.int32:
            raise ValueError("dtype must be int32")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        self.BinomialN(out.size, out, ntot, prob)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.int32)
        self.BinomialN(int(sz), ret, ntot, prob)
        return ret.reshape(size)

def _TRandom_BreitWigner(self, mean=0, gamma=1, size=None, out=None):

    import ROOT
    import numpy

    if size == None and out == None:
        return self._BreitWigner(mean, gamma)
    if out != None:
        if out.dtype != numpy.float64:
            raise ValueError("dtype must be float64")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        self.BreitWignerN(out.size, out, mean, gamma)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.float64)
        self.BreitWignerN(int(sz), ret, mean, gamma)
        return ret.reshape(size)

def _TRandom_Exp(self, tau, size=None, out=None):

    import ROOT
    import numpy

    if size == None and out == None:
        return self._Exp(tau)
    if out != None:
        if out.dtype != numpy.float64:
            raise ValueError("dtype must be float64")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        self.ExpN(out.size, out, tau)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.float64)
        self.ExpN(int(sz), ret, tau)
        return ret.reshape(size)

def _TRandom_Gaus(self, mean=0, sigma=1, size=None, out=None):

    import ROOT
    import numpy

    if size == None and out is None:
        return self._Gaus(mean, sigma)
    if out is not None:
        if out.dtype != numpy.float64:
            raise ValueError("dtype must be float64")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        self.GausN(out.size, out, mean, sigma)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.float64)
        self.GausN(int(sz), ret, mean, sigma)
        return ret.reshape(size)

def _TRandom_Integer(self, imax, size=None, out=None):

    import ROOT
    import numpy

    if size == None and out == None:
        return self._Integer(imax)
    if out != None:
        if out.dtype != numpy.uint32:
            raise ValueError("dtype must be uint32")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        self.IntegerN(out.size, out, imax)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.uint32)
        self.IntegerN(int(sz), ret, imax)
        return ret.reshape(size)

def _TRandom_Landau(self, mean=0, sigma=1, size=None, out=None):

    import ROOT
    import numpy

    if size == None and out == None:
        return self._Landau(mean, sigma)
    if out != None:
        if out.dtype != numpy.float64:
            raise ValueError("dtype must be float64")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        self.LandauN(out.size, out, mean, sigma)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.float64)
        self.LandauN(int(sz), ret, mean, sigma)
        return ret.reshape(size)

def _TRandom_Poisson(self, mean, size=None, out=None):

    import ROOT
    import numpy

    if size == None and out == None:
        return self._Poisson(mean)
    if out != None:
        if out.dtype != numpy.ulonglong:
            raise ValueError("dtype must be ulonglong")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        self.PoissonN(out.size, out, mean)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.ulonglong)
        self.PoissonN(int(sz), ret, mean)
        return ret.reshape(size)

def _TRandom_PoissonD(self, mean, size=None, out=None):

    import ROOT
    import numpy

    if size == None and out == None:
        return self._PoissonD(mean)
    if out != None:
        if out.dtype != numpy.float64:
            raise ValueError("dtype must be float64")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        self.PoissonDN(out.size, out, mean)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.float64)
        self.PoissonDN(int(sz), ret, mean)
        return ret.reshape(size)

def _TRandom_Uniform(self, x1=0, x2=1, size=None, out=None):
    import ROOT
    import numpy

    if size == None and out == None:
        return self._Uniform(x2) if x1 == 0 else self._Uniform(x1, x2)
    if out != None:
        if out.dtype != numpy.float64:
            raise ValueError("dtype must be float64")
        sz = numpy.shape(out)
        if size != None and size != sz:
            raise ValueError("size must match out.shape when used together")
        out = out.reshape(-1)
        if x1 == 0:
            self.UniformN(out.size, out, x2)
        else:
            self.UniformN(out.size, out, x1, x2)
        out = out.reshape(sz)
        return None
    else:
        sz = numpy.prod(size)
        ret = numpy.empty(shape=sz, dtype=numpy.float64)
        if x1 == 0:
            self.UniformN(int(sz), ret, x2)
        else:
            self.UniformN(int(sz), ret, x1, x2)
        return ret.reshape(size)


@pythonization('TRandom')
def pythonize_trandom(klass):   

    # Pythonizations for TRandom::Binomial
    klass._Binomial = klass.Binomial
    klass.Binomial = _TRandom_Binomial

    # Pythonizations for TRandom::BreitWigner
    klass._BreitWigner = klass.BreitWigner
    klass.BreitWigner = _TRandom_BreitWigner

    # Pythonizations for TRandom::Exp
    klass._Exp = klass.Exp
    klass.Exp = _TRandom_Exp

    # Pythonizations for TRandom::Gaus
    klass._Gaus = klass.Gaus
    klass.Gaus = _TRandom_Gaus

    # Pythonizations for TRandom::Integer
    klass._Integer = klass.Integer
    klass.Integer = _TRandom_Integer

    # Pythonizations for TRandom::Landau
    klass._Landau = klass.Landau
    klass.Landau = _TRandom_Landau

    # Pythonizations for TRandom::Poisson
    klass._Poisson = klass.Poisson
    klass.Poisson = _TRandom_Poisson

    # Pythonizations for TRandom::PoissonD
    klass._PoissonD = klass.PoissonD
    klass.PoissonD = _TRandom_PoissonD

    # Pythonizations for TRandom::Uniform
    klass._Uniform = klass.Uniform
    klass.Uniform = _TRandom_Uniform

