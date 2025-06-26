// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Minuit2_MnApplication
#define ROOT_Minuit2_MnApplication

#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnStrategy.h"

#include <vector>

namespace ROOT {

namespace Minuit2 {

class FunctionMinimum;
class MinuitParameter;
class MnMachinePrecision;
class ModularFunctionMinimizer;
class FCNBase;

//___________________________________________________________________________
/**
    application interface class for minimizers (migrad, simplex, Minimize,
    Scan)
    User normally instantiates the derived class like ROOT::Minuit2::MnMigrad
    for using Migrad for minimization
 */

class MnApplication {

public:
   /// constructor from non-gradient functions
   MnApplication(const FCNBase &fcn, const MnUserParameterState &state, const MnStrategy &stra, unsigned int nfcn = 0);

   virtual ~MnApplication() {}

   /**
      Minimize the function
      @param maxfcn : max number of function calls (if = 0) default is used which is set to
                     200 + 100 * npar + 5 * npar**2
      @param tolerance : value used for terminating iteration procedure.
             For example, MIGRAD will stop iterating when edm (expected distance from minimum) will be:
             edm < tolerance * 10**-3
             Default value of tolerance used is 0.1
   */
   virtual FunctionMinimum operator()(unsigned int maxfcn = 0, double tolerance = 0.1);

   virtual ModularFunctionMinimizer &Minimizer() = 0;
   virtual const ModularFunctionMinimizer &Minimizer() const = 0;

   const MnMachinePrecision &Precision() const { return fState.Precision(); }
   MnUserParameterState &State() { return fState; }
   const MnUserParameterState &State() const { return fState; }
   const MnUserParameters &Parameters() const { return fState.Parameters(); }
   const MnUserCovariance &Covariance() const { return fState.Covariance(); }
   virtual const FCNBase &Fcnbase() const { return fFCN; }
   const MnStrategy &Strategy() const { return fStrategy; }
   unsigned int NumOfCalls() const { return fNumCall; }

protected:
   const FCNBase &fFCN;
   MnUserParameterState fState;
   MnStrategy fStrategy;
   unsigned int fNumCall;
};

} // namespace Minuit2

} // namespace ROOT

#endif // ROOT_Minuit2_MnApplication
