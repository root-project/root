## \file
## \ingroup tutorial_math
## \notebook
## Example of using Python functions as inputs to numerical algorithms
## using the ROOT Functor class. 
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Lorenzo Moneta

import array
import sys  # TEMP-DIAG: r1: see below

import ROOT

try:
    import numpy as np
except Exception as e:
    print("Failed to import numpy:", e)
    exit()

## example 1D function

def f(x):
   return x*x -1

func = ROOT.Math.Functor1D(f)

#example of using the integral

print("Use Functor1D for wrapping one-dimensional function and compute integral of f(x) = x^2-1")

ig = ROOT.Math.Integrator()
ig.SetFunction(func)

value = ig.Integral(0, 3)
print("integral-1D value = ", value)
expValue = 6
if (not ROOT.TMath.AreEqualRel(value, expValue, 1.E-15)) :
   print("Error computing integral - computed value - different than expected, diff = ", value - expValue)
   
# example multi-dim function

print("\n\nUse Functor for wrapping a multi-dimensional function, the Rosenbrock Function r(x,y) and find its minimum")

def RosenbrockFunction(xx):
  x = xx[0]
  y = xx[1]
  tmp1 = y-x*x
  tmp2 = 1-x
  return 100*tmp1*tmp1+tmp2*tmp2


func2D = ROOT.Math.Functor(RosenbrockFunction,2)

### minimize multi-dim function using fitter class

fitter = ROOT.Fit.Fitter()
#use a numpy array to pass initial parameter array 
initialParams = np.array([0.,0.], dtype='d')
fitter.FitFCN(func2D, initialParams)
fitter.Result().Print(ROOT.std.cout)
if (not ROOT.TMath.AreEqualRel(fitter.Result().Parameter(0), 1, 1.E-3) or not ROOT.TMath.AreEqualRel(fitter.Result().Parameter(1), 1, 1.E-3)) :
   print("Error minimizing Rosenbrock function ")

## example 1d grad function
## derivative of f(x)= x**2-1

print("\n\nUse GradFunctor1D for making a function object implementing f(x) and f'(x)")

def g(x): return 2 * x

# TEMP-DIAG: r1: on Windows the GradFunctor1D below crashes in the compiled DoEval(). It is the first class with
# a virtual base that the interpreter constructs here: compare its layout as seen by MSVC and by the interpreter.
if sys.platform == "win32":
    import ctypes
    import struct

    _jc_psz = ctypes.sizeof(ctypes.c_void_p)

    def _jc(*args):
        print("[jc-TEMP]", *args, file=sys.stderr, flush=True)

    def _jc_where(addr):
        # " <module+0xRVA>" if addr points into a loaded module
        if sys.platform != "win32" or addr < 0x10000:
            return ""
        try:
            k32 = ctypes.WinDLL("kernel32")
            k32.GetModuleHandleExW.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            k32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
            h = ctypes.c_void_p()
            # GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT
            if not k32.GetModuleHandleExW(0x4 | 0x2, ctypes.c_void_p(addr), ctypes.byref(h)) or not h.value:
                return ""
            buf = ctypes.create_unicode_buffer(260)
            k32.GetModuleFileNameW(h, buf, 260)
            return " <%s+0x%x>" % (buf.value.rsplit("\\", 1)[-1], addr - h.value)
        except Exception as e:
            return " <where failed: %r>" % (e,)

    def _jc_words(addr, nbytes):
        data = ctypes.string_at(addr, nbytes // _jc_psz * _jc_psz)
        return struct.unpack("<%d%s" % (len(data) // _jc_psz, "Q" if _jc_psz == 8 else "I"), data)

    def _jc_dump(tag, addr, size, vbase=True):
        # raw object memory; with vbase, word 0 is taken as vbptr to also show the vbtable and the vbase's vftable
        _jc("%s: addr=0x%x size=%d" % (tag, addr, size))
        try:
            words = _jc_words(addr, size)
            for i, w in enumerate(words):
                _jc("   +%3d: 0x%0*x%s" % (i * _jc_psz, 2 * _jc_psz, w, _jc_where(w)))
        except Exception as e:
            _jc("   dump failed: %r" % (e,))
            return
        if not vbase:
            return
        try:
            vbt = struct.unpack("<ii", ctypes.string_at(words[0], 8))
            _jc("   vbtable: %r" % (vbt,))
            if 0 < vbt[1] <= size - _jc_psz:
                vfptr = _jc_words(addr + vbt[1], _jc_psz)[0]
                (vtordisp,) = struct.unpack("<i", ctypes.string_at(addr + vbt[1] - 4, 4))
                _jc("   vbase at +%d: vfptr=0x%x%s, int32 before it=%d" % (vbt[1], vfptr, _jc_where(vfptr), vtordisp))
                for i, w in enumerate(_jc_words(vfptr, 8 * _jc_psz)):
                    _jc("      vftable[%d]: 0x%x%s" % (i, w, _jc_where(w)))
        except Exception as e:
            _jc("   vtable dump failed: %r" % (e,))

    def _jc_step(what, func):
        _jc("step:", what)
        try:
            _jc("   ->", func())
        except Exception as e:
            _jc("   raised %r" % (e,))

    _jc_calc = ROOT.gInterpreter.Calc

    def _jc_ptr(expr):
        return _jc_calc("(Longptr_t)(%s)" % expr) & ((1 << (8 * _jc_psz)) - 1)

    def _jc_sizes(name):
        cl = ROOT.TClass.GetClass(name)
        return cl.Size(), _jc_calc("sizeof(%s)" % name), cl.HasDictionary()

    # 1. sizeof according to the dictionary (compiled) and to the interpreter
    for _jc_name in [
        "ROOT::Math::IBaseFunctionOneDim",
        "ROOT::Math::IGradientFunctionOneDim",
        "ROOT::Math::Functor1D",
        "ROOT::Math::GradFunctor1D",
        "ROOT::Math::Functor",
        "ROOT::Math::GradFunctor",
        "ROOT::Math::IBaseParam",
        "ROOT::Math::IParametricFunctionOneDim",
        "ROOT::Math::IParametricGradFunctionOneDim",
        "ROOT::Math::WrappedTF1",
        "ROOT::Math::WrappedMultiTF1",
        "TF1",
        "TH1D",
        "TRandom3",
    ]:
        try:
            _jc("sizeof %s: compiled=%d interpreter=%d hasDict=%d" % ((_jc_name,) + _jc_sizes(_jc_name)))
        except Exception as e:
            _jc("sizeof %s failed: %r" % (_jc_name, e))

    # 2. default-constructed objects: by compiled code (dictionary), by the interpreter, by Python
    for _jc_name in ["ROOT::Math::Functor1D", "ROOT::Math::GradFunctor1D", "ROOT::Math::GradFunctor"]:
        try:
            _jc_cs, _jc_is, _ = _jc_sizes(_jc_name)
            _jc_vb = "Grad" in _jc_name
            _jc_dump("compiled    %s()" % _jc_name, _jc_ptr('TClass::GetClass("%s")->New()' % _jc_name), _jc_cs, _jc_vb)
            _jc_dump("interpreter %s()" % _jc_name, _jc_ptr("new %s()" % _jc_name), _jc_is, _jc_vb)
        except Exception as e:
            _jc("default objects of %s failed: %r" % (_jc_name, e))
    _jc_g0 = ROOT.Math.GradFunctor1D()
    _jc_dump("python      GradFunctor1D()", ROOT.addressof(_jc_g0), _jc_calc("sizeof(ROOT::Math::GradFunctor1D)"))

    # 3. pure C++: interpreter-constructed GradFunctor1D, used by the interpreter and then by compiled code
    ROOT.gInterpreter.Declare("""
    #include "Math/Functor.h"
    #include "Math/Integrator.h"
    #include <functional>
    namespace JcTemp {
    ROOT::Math::GradFunctor1D *gF = nullptr;
    void Make() {
       gF = new ROOT::Math::GradFunctor1D(std::function<double(double)>([](double x) { return x * x - 1; }),
                                          std::function<double(double)>([](double x) { return 2 * x; }));
    }
    double Integral() {
       ROOT::Math::IntegratorOneDim ig;
       ig.SetFunction(static_cast<const ROOT::Math::IGenFunction &>(*gF));
       return ig.Integral(0, 3);
    }
    double FdF() {
       const ROOT::Math::IGradientFunctionOneDim &ref = *gF;
       double f = 0, df = 0;
       ref.FdF(3., f, df);
       return f + df;
    }
    }
    """)
    _jc_step("C++ construct", lambda: ROOT.JcTemp.Make())
    _jc_dump("interpreter GradFunctor1D(f, g)", _jc_ptr("JcTemp::gF"), _jc_calc("sizeof(ROOT::Math::GradFunctor1D)"))
    _jc_step("C++ HasGradient() [expect 1]", lambda: _jc_calc("(Longptr_t)JcTemp::gF->HasGradient()"))
    _jc_step("C++ operator()(3) [expect 8]", lambda: _jc_calc("(Longptr_t)(*JcTemp::gF)(3.)"))
    _jc_step("C++ Derivative(3) [expect 6]", lambda: _jc_calc("(Longptr_t)JcTemp::gF->Derivative(3.)"))
    _jc_step("C++ virtual FdF(3) [expect 14]", lambda: ROOT.JcTemp.FdF())
    _jc_step("C++ compiled Integrator [expect 6]", lambda: ROOT.JcTemp.Integral())

    # 4. the object of the tutorial: constructed through Python with Python callables
    _jc_gf = ROOT.Math.GradFunctor1D(f, g)
    _jc_dump("python      GradFunctor1D(f, g)", ROOT.addressof(_jc_gf), _jc_calc("sizeof(ROOT::Math::GradFunctor1D)"))
    _jc_f1 = ROOT.Math.Functor1D(f)
    _jc_dump("python      Functor1D(f)", ROOT.addressof(_jc_f1), _jc_calc("sizeof(ROOT::Math::Functor1D)"), False)
    _jc_step("python HasGradient() [expect True]", lambda: _jc_gf.HasGradient())
    _jc_step("python operator()(3) [expect 8]", lambda: _jc_gf(3.0))
    _jc_step("python Derivative(3) [expect 6]", lambda: _jc_gf.Derivative(3.0))
    _jc("diagnostics done, continuing with the tutorial")

# GSL_Newton is part of ROOT's MathMore library, which is not always active.
# Let's therefore run this in a try block:
try:
    gradFunc = ROOT.Math.GradFunctor1D(f, g)
    rf = ROOT.Math.RootFinder(ROOT.Math.RootFinder.kGSL_NEWTON)
    rf.SetFunction(gradFunc, 3)
    rf.Solve()
    value = rf.Root()
    print("Found root value x0 : f(x0) = 0  :  ", value)
    if value != 1:
        print("Error finding a ROOT of function f(x)=x^2-1")
except Exception as e:
    print(e)


print("\n\nUse GradFunctor for making a function object implementing f(x,y) and df(x,y)/dx and df(x,y)/dy")

def RosenbrockDerivatives(xx, icoord):
  x = xx[0]
  y = xx[1]
  #derivative w.r.t x 
  if (icoord == 0) :
    return 2*(200*x*x*x-200*x*y+x-1)
  else : 
    return 200 * (y - x * x)
    
gradFunc2d = ROOT.Math.GradFunctor(RosenbrockFunction, RosenbrockDerivatives, 2)

fitter = ROOT.Fit.Fitter()
#here we use a python array to pass initial parameters
initialParams = array.array('d',[0.,0.])
fitter.FitFCN(gradFunc2d, initialParams)
fitter.Result().Print(ROOT.std.cout)
if (not ROOT.TMath.AreEqualRel(fitter.Result().Parameter(0), 1, 1.E-3) or not ROOT.TMath.AreEqualRel(fitter.Result().Parameter(1), 1, 1.E-3)) :
   print("Error minimizing Rosenbrock function ")
