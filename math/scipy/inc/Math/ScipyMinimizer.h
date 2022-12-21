// @(#)root/math/scipy:$Id$
// Author: Omar.Zapata@cern.ch 2022

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

#include "Math/GenAlgoOptions.h"

#include "Rtypes.h"
#include "TString.h"

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
 * \class ScipyMinimizer
 * ScipyMinimizer class.
 * Implementation for Scipy ... TODO

   See <A HREF="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html">Scipy doc</A>
   from more info on the Scipy minimization algorithms.

   @ingroup MultiMin
*/

class ScipyMinimizer : public BasicMinimizer {
private:
   PyObject *fMinimize;
   PyObject *fTarget;
   PyObject *fJacobian;
   GenAlgoOptions fExtraOpts;

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
      Python eval function
    */
   PyObject *Eval(TString code);

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

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   ScipyMinimizer(const ScipyMinimizer &) : BasicMinimizer() {}
   void SetAlgoExtraOptions();

public:
   /// set the function to minimize
   // virtual void SetFunction(const ROOT::Math::IMultiGenFunction &func);

   /// set the function to minimize
   // virtual void SetFunction(const ROOT::Math::IMultiGradFunction &func) { BasicMinimizer::SetFunction(func); }

   /// method to perform the minimization
   virtual bool Minimize() override;

   template <class T>
   void SetExtraOption(const char *key, T value)
   {
      fExtraOpts.SetValue(key, value);
   }

protected:
   ClassDef(ScipyMinimizer, 0) //
};

} // end namespace Experimental

} // end namespace Math

} // end namespace ROOT

#endif /* ROOT_Math_ScipyMinimizer */
