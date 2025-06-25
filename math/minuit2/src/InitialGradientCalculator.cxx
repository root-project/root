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
#include "Minuit2/MnPrint.h"

#include <cmath>

namespace ROOT {

namespace Minuit2 {

/// Initial rough estimate of the gradient using the parameter step size.
FunctionGradient
calculateInitialGradient(const MinimumParameters &par, const MnUserTransformation &trafo, double errorDef)
{
   assert(par.IsValid());

   unsigned int n = trafo.VariableParameters();
   assert(n == par.Vec().size());

   MnPrint print("InitialGradientCalculator");

   print.Debug("Calculating initial gradient at point", par.Vec());

   MnAlgebraicVector gr(n), gr2(n), gst(n);

   for (unsigned int i = 0; i < n; i++) {
      unsigned int exOfIn = trafo.ExtOfInt(i);

      double var = par.Vec()(i);
      double werr = trafo.Parameter(exOfIn).Error();
      double save1 = trafo.Int2ext(i, var);
      double save2 = save1 + werr;
      if (trafo.Parameter(exOfIn).HasLimits()) {
         if (trafo.Parameter(exOfIn).HasUpperLimit() && save2 > trafo.Parameter(exOfIn).UpperLimit())
            save2 = trafo.Parameter(exOfIn).UpperLimit();
      }
      double var2 = trafo.Ext2int(exOfIn, save2);
      double vplu = var2 - var;
      save2 = save1 - werr;
      if (trafo.Parameter(exOfIn).HasLimits()) {
         if (trafo.Parameter(exOfIn).HasLowerLimit() && save2 < trafo.Parameter(exOfIn).LowerLimit())
            save2 = trafo.Parameter(exOfIn).LowerLimit();
      }
      var2 = trafo.Ext2int(exOfIn, save2);
      double vmin = var2 - var;
      double gsmin = 8. * trafo.Precision().Eps2() * (std::fabs(var) + trafo.Precision().Eps2());
      // protect against very small step sizes which can cause dirin to zero and then nan values in grd
      double dirin = std::max(0.5 * (std::fabs(vplu) + std::fabs(vmin)), gsmin);
      double g2 = 2.0 * errorDef / (dirin * dirin);
      double gstep = std::max(gsmin, 0.1 * dirin);
      double grd = g2 * dirin;
      if (trafo.Parameter(exOfIn).HasLimits()) {
         if (gstep > 0.5)
            gstep = 0.5;
      }
      gr(i) = grd;
      gr2(i) = g2;
      gst(i) = gstep;

      print.Trace("Computed initial gradient for parameter", trafo.Name(exOfIn), "value", var, "[", vmin, ",", vplu,
                  "]", "dirin", dirin, "grd", grd, "g2", g2);
   }

   return FunctionGradient(gr, gr2, gst);
}

} // namespace Minuit2

} // namespace ROOT
