// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/InitialGradientCalculator.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MnStrategy.h"

#include <math.h>

//#define DEBUG

#if defined(DEBUG) || defined(WARNINGMSG)
#include "Minuit2/MnPrint.h"
#endif



namespace ROOT {

   namespace Minuit2 {


FunctionGradient InitialGradientCalculator::operator()(const MinimumParameters& par) const {
   // initial rough  estimate of the gradient using the parameter step size

   assert(par.IsValid());

   unsigned int n = Trafo().VariableParameters();
   assert(n == par.Vec().size());

#ifdef DEBUG
   std::cout << "Initial gradient calculator - params " << par.Vec() << std::endl;
#endif

   MnAlgebraicVector gr(n), gr2(n), gst(n);
   std::cout << "initial gradient minuit: " << std::endl;

   for(unsigned int i = 0; i < n; i++) {
      unsigned int exOfIn = Trafo().ExtOfInt(i);

      double var = par.Vec()(i);
      std::cout << "var: " << var << std::endl;

      double werr = Trafo().Parameter(exOfIn).Error();
      std::cout << "werr: " << werr << std::endl;
      double sav = Trafo().Int2ext(i, var);
      std::cout << "sav: " << sav << std::endl;

      double sav2 = sav + werr;
      std::cout << "sav2: " << sav2 << std::endl;
      if(Trafo().Parameter(exOfIn).HasLimits()) {
         if(Trafo().Parameter(exOfIn).HasUpperLimit() &&
            sav2 > Trafo().Parameter(exOfIn).UpperLimit())
            sav2 = Trafo().Parameter(exOfIn).UpperLimit();
      }
      std::cout << "sav2: " << sav2 << std::endl;
      double var2 = Trafo().Ext2int(exOfIn, sav2);
      std::cout << "var2: " << var2 << std::endl;
      double vplu = var2 - var;
      std::cout << "vplu: " << vplu << std::endl;
      sav2 = sav - werr;
      std::cout << "sav2: " << sav2 << std::endl;
      if(Trafo().Parameter(exOfIn).HasLimits()) {
         if(Trafo().Parameter(exOfIn).HasLowerLimit() &&
            sav2 < Trafo().Parameter(exOfIn).LowerLimit())
            sav2 = Trafo().Parameter(exOfIn).LowerLimit();
      }
      std::cout << "sav2: " << sav2 << std::endl;
      var2 = Trafo().Ext2int(exOfIn, sav2);
      std::cout << "var2: " << var2 << std::endl;
      double vmin = var2 - var;
      std::cout << "vmin: " << vmin << std::endl;
      double gsmin = 8.*Precision().Eps2()*(fabs(var) + Precision().Eps2());
      std::cout << "gsmin: " << gsmin << std::endl;
      // protect against very small step sizes which can cause dirin to zero and then nan values in grd
      double dirin = std::max(0.5*(fabs(vplu) + fabs(vmin)),  gsmin );
      std::cout << "dirin: " << dirin << std::endl;
      double g2 = 2.0*fFcn.ErrorDef()/(dirin*dirin);
      double gstep = std::max(gsmin, 0.1*dirin);
      std::cout << "gstep: " << gstep << std::endl;
      double grd = g2*dirin;
      if(Trafo().Parameter(exOfIn).HasLimits()) {
         if(gstep > 0.5) gstep = 0.5;
      }
      gr(i) = grd;
      gr2(i) = g2;
      gst(i) = gstep;

      std::cout << "fGrd[" << i <<"] = " << gr(i) << "\t";
      std::cout << "fG2[" << i <<"] = " << gr2(i) << "\t";
      std::cout << "fGstep[" << i <<"] = " << gst(i) << std::endl;

#ifdef DEBUG
      std::cout << "computing initial gradient for parameter " << Trafo().Name(exOfIn) << " value = " << var
                << " [ " << vmin << " , " << vplu << " ] " << "dirin " <<  dirin << " grd " << grd << " g2 " << g2 << std::endl;
#endif

   }

   return FunctionGradient(gr, gr2, gst);
}

FunctionGradient InitialGradientCalculator::operator()(const MinimumParameters& par, const FunctionGradient&) const {
   // Base class interface
   return (*this)(par);
}

const MnMachinePrecision& InitialGradientCalculator::Precision() const {
   // return precision (is set in trasformation class)
   return fTransformation.Precision();
}

unsigned int InitialGradientCalculator::Ncycle() const {
   // return ncyles (from Strategy)
   return Strategy().GradientNCycles();
}

double InitialGradientCalculator::StepTolerance() const {
   // return Gradient step tolerance (from Strategy)
   return Strategy().GradientStepTolerance();
}

double InitialGradientCalculator::GradTolerance() const {
   // return Gradient tolerance
   return Strategy().GradientTolerance();
}


   }  // namespace Minuit2

}  // namespace ROOT
