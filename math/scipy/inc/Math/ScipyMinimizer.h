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
   int  PyIsInitialized();

   /*
      Python finalization
    */
   void PyFinalize();
   void PyRunString(TString code, TString errorMessage="Failed to run python code", int start=Py_single_input);
   void LoadWrappers();
private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   ScipyMinimizer(const ScipyMinimizer &) : BasicMinimizer() {}


public:
   /// set the function to minimize
   //virtual void SetFunction(const ROOT::Math::IMultiGenFunction &func);

   /// set the function to minimize
   //virtual void SetFunction(const ROOT::Math::IMultiGradFunction &func) { BasicMinimizer::SetFunction(func); }

   /// method to perform the minimization
   virtual bool Minimize();

   /// return expected distance reached from the minimum
   virtual double Edm() const { return 0; } // not impl. }

   /// minimizer provides error and error matrix
   virtual bool ProvidesError() const { return false; }

   /// return errors at the minimum
   virtual const double *Errors() const { return 0; }

   /** return covariance matrices elements
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */
   virtual double CovMatrix(unsigned int, unsigned int) const { return 0; }
protected:
   ClassDef(ScipyMinimizer, 0) //
};

} // end namespace Experimental

} // end namespace Math

} // end namespace ROOT

#endif /* ROOT_Math_ScipyMinimizer */
