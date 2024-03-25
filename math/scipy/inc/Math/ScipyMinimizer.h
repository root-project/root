// @(#)root/math/scipy:$Id$
// Author: Omar.Zapata@cern.ch 2023

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Math_ScipyMinimizer
#define ROOT_Math_ScipyMinimizer

#include "Math/Minimizer.h"
#include "Math/MinimizerOptions.h"

#include "Math/IFunctionfwd.h"

#include "Math/IParamFunctionfwd.h"

#include "Math/BasicMinimizer.h"

#include "Rtypes.h"
#include "TString.h"

#include <functional>
#include <vector>
#include <map>

#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#define Py_single_input 256
#endif

namespace ROOT {

namespace Math {

class GenAlgoOptions;

namespace Experimental {
/**
   enumeration specifying the types of Scipy solvers
   @ingroup MultiMin
*/

//_____________________________________________________________________________________
/**
  \class ScipyMinimizer
  ScipyMinimizer class.
  Scipy minimizer implementation using Python C API that supports several methods such as
  Nelder-Mead, L-BFGS-B, Powell, CG, BFGS, TNC, COBYLA, SLSQP, trust-constr,
  Newton-CG, dogleg, trust-ncg, trust-exact and trust-krylov.

  It supports the Jacobian (Gradients), Hessian and bounds for the variables.

  Support for constraint functions will be implemented in the next releases.
  You can find a macro example in the folder $ROOTSYS/tutorial/fit/scipy.C

  To provide extra options to the minimizer, you can use the class GenAlgoOptions
  and the method SetExtraOptions().
  Example:
   ```
   ROOT::Math::GenAlgoOptions l_bfgs_b_opt;
   l_bfgs_b_opt.SetValue("gtol", 1e-3);
   l_bfgs_b_opt.SetValue("ftol", 1e7);
   minimizer->SetExtraOptions(l_bfgs_b_opt);
   ```

  See <A HREF="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html">Scipy doc</A>
  from more info on the Scipy minimization algorithms.

   @ingroup MultiMin
*/

class ScipyMinimizer : public BasicMinimizer {
private:
   PyObject *fMinimize;
   PyObject *fTarget;
   PyObject *fJacobian;
   PyObject *fHessian;
   PyObject *fBoundsMod;
   PyObject *fConstraintsList; /// contraints functions
   GenAlgoOptions *fExtraOpts;
   std::function<bool(const std::vector<double> &, double *)> fHessianFunc;
   unsigned int fConstN;
   unsigned int fCalls;

protected:
   PyObject *fGlobalNS;
   PyObject *fLocalNS;

public:
   /**
      Default constructor
   */
   ScipyMinimizer();
   /**
      Constructor with a string giving name of algorithm
   */
   ScipyMinimizer(const char *type);

   /**
      Destructor
   */
   virtual ~ScipyMinimizer();

   /**
      Python initialization
    */
   void PyInitialize();

   /**
      Checks if Python was initialized
    */
   int PyIsInitialized();

   /*
      Python finalization
    */
   void PyFinalize();
   /*
      Python code execution
    */
   void PyRunString(TString code, TString errorMessage = "Failed to run python code", int start = Py_single_input);

   /*
      Number of function calls
    */
   virtual unsigned int NCalls() const override;

   /*
      Method to add Constraint function,
      multiples constraints functions can be added.
      type have to be a string "eq" or "ineq" where
      eq (means Equal, then fun() = 0)
      ineq (means that, it is to be non-negative. fun() >=0)
      https://kitchingroup.cheme.cmu.edu/f19-06623/13-constrained-optimization.html
   */
   virtual void AddConstraintFunction(std::function<double(const std::vector<double> &)>, std::string type);

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   ScipyMinimizer(const ScipyMinimizer &) : BasicMinimizer() {}

   /**
      Get extra options from IOptions
   */
   void SetExtraOptions();

public:
   /// method to perform the minimization
   virtual bool Minimize() override;

   virtual void SetHessianFunction(std::function<bool(const std::vector<double> &, double *)>) override;

protected:
   ClassDef(ScipyMinimizer, 0) //
};

} // end namespace Experimental

} // end namespace Math

} // end namespace ROOT

#endif /* ROOT_Math_ScipyMinimizer */
