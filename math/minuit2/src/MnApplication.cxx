// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnApplication.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/ModularFunctionMinimizer.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnPrint.h"

namespace ROOT {

namespace Minuit2 {

// constructor from non-gradient functions
MnApplication::MnApplication(const FCNBase &fcn, const MnUserParameterState &state, const MnStrategy &stra,
                             unsigned int nfcn)
   : fFCN(fcn), fState(state), fStrategy(stra), fNumCall(nfcn)
{
}

FunctionMinimum MnApplication::operator()(unsigned int maxfcn, double toler)
{
   // constructor from macfcn calls and tolerance
   MnPrint print("MnApplication");

   assert(fState.IsValid());
   unsigned int npar = State().VariableParameters();
   //   assert(npar > 0);
   if (maxfcn == 0)
      maxfcn = 200 + 100 * npar + 5 * npar * npar;

   const FCNBase &fcn = Fcnbase();


   if (npar == 0) {
      double fval = fcn(fState.Params());
      print.Info("Function has zero parameters - returning current function value - ",fval);
      // create a valid Minuit-Parameter object with just the function value
      MinimumParameters mparams(fval, MinimumParameters::MnValid);
      MinimumState mstate(mparams, 0., 1 );
      return FunctionMinimum( MinimumSeed(mstate, fState.Trafo()), fcn.Up());
   }

   FunctionMinimum min = Minimizer().Minimize(fcn, fState, fStrategy, maxfcn, toler);

   fNumCall += min.NFcn();
   fState = min.UserState();

   const std::vector<ROOT::Minuit2::MinimumState> &iterationStates = min.States();
   print.Debug("State resulting from Migrad after", iterationStates.size(), "iterations:", fState);

   print.Debug([&](std::ostream &os) {
      for (unsigned int i = 0; i < iterationStates.size(); ++i) {
         // std::cout << iterationStates[i] << std::endl;
         const ROOT::Minuit2::MinimumState &st = iterationStates[i];
         os << "\n----------> Iteration " << i << '\n';
         int pr = os.precision(18);
         os << "            FVAL = " << st.Fval() << " Edm = " << st.Edm() << " Nfcn = " << st.NFcn() << '\n';
         os.precision(pr);
         os << "            Error matrix change = " << st.Error().Dcovar() << '\n';
         os << "            Internal parameters : ";
         for (int j = 0; j < st.size(); ++j)
            os << " p" << j << " = " << st.Vec()(j);
      }
   });

   return min;
}

} // namespace Minuit2

} // namespace ROOT
