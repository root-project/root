// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/NegativeG2LineSearch.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/GradientCalculator.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/VariableMetricEDMEstimator.h"

#include <cmath>
//#define DEBUG
#ifdef DEBUG
#include <iostream>
#endif

namespace ROOT {

   namespace Minuit2 {




MinimumState NegativeG2LineSearch::operator()(const MnFcn& fcn, const MinimumState& st, const  GradientCalculator& gc, const MnMachinePrecision& prec) const {

//   when the second derivatives are negative perform a  line search  along Parameter which gives
//   negative second derivative and magnitude  equal to the Gradient step size.
//   Recalculate the gradients for all the Parameter after the correction and
//   continue iteration in case the second derivatives are still negative
//

   bool negG2 = HasNegativeG2(st.Gradient(), prec);
   if(!negG2) return st;

   unsigned int n = st.Parameters().Vec().size();
   FunctionGradient dgrad = st.Gradient();
   MinimumParameters pa = st.Parameters();
   bool iterate = false;
   unsigned int iter = 0;
   do {
      iterate = false;
      for(unsigned int i = 0; i < n; i++) {

#ifdef DEBUG
         std::cout << "negative G2 - iter " << iter << " param " << i << "   " << pa.Vec()(i) << "  grad2 " << dgrad.G2()(i) << " grad " << dgrad.Vec()(i)
                   << " grad step " << dgrad.Gstep()(i) << " step size " << pa.Dirin()(i) << std::endl;
#endif
         if(dgrad.G2()(i) <= 0) {

            // check also the gradient (if it is zero ) I can skip the param)

            if ( std::fabs(dgrad.Vec()(i) ) < prec.Eps() && std::fabs(dgrad.G2()(i) ) < prec.Eps() ) continue;
            //       if(dgrad.G2()(i) < prec.Eps()) {
            // do line search if second derivative negative
            MnAlgebraicVector step(n);
            MnLineSearch lsearch;

            if ( dgrad.Vec()(i) < 0)
               step(i) = dgrad.Gstep()(i); //*dgrad.Vec()(i);
            else
               step(i) = - dgrad.Gstep()(i); // *dgrad.Vec()(i);

            double gdel = step(i)*dgrad.Vec()(i);

            // if using sec derivative information
            // double g2del = step(i)*step(i) * dgrad.G2()(i);
            bool debugLS = false;

#ifdef DEBUG
            std::cout << "step(i) " << step(i) << " gdel " << gdel << std::endl;
//            std::cout << " g2del " << g2del << std::endl;
            debugLS = true;
#endif
            MnParabolaPoint pp = lsearch(fcn, pa, step, gdel, prec,debugLS);



#ifdef DEBUG
            std::cout << "\nLine search result " << pp.X() << " f(0)  " << pa.Fval() << " f(1) " << pp.Y() << std::endl;
#endif

            step *= pp.X();
            pa = MinimumParameters(pa.Vec() + step, pp.Y());

            dgrad = gc(pa, dgrad);

#ifdef DEBUG
            std::cout << "Line search - iter" << iter << " param " << i << "   " << pa.Vec()(i) << " step " << step(i) << " new grad2 " << dgrad.G2()(i) << " new grad " <<  dgrad.Vec()(i) << " grad step " << dgrad.Gstep()(i) << std::endl;
#endif

            iterate = true;
            break;
            }
         }
      } while(iter++ < 2*n && iterate);

   MnAlgebraicSymMatrix mat(n);
   for(unsigned int i = 0; i < n; i++)
      mat(i,i) = (fabs(dgrad.G2()(i)) > prec.Eps2() ? 1./dgrad.G2()(i) : 1.);

   MinimumError err(mat, 1.);
   double edm = VariableMetricEDMEstimator().Estimate(dgrad, err);

   return MinimumState(pa, err, dgrad, edm, fcn.NumOfCalls());
}

bool NegativeG2LineSearch::HasNegativeG2(const FunctionGradient& grad, const MnMachinePrecision& /*prec */ ) const {
   // check if function gradient has any component which is neegative

   for(unsigned int i = 0; i < grad.Vec().size(); i++)

      if(grad.G2()(i) <= 0 ) {

         return true;
      }

   return false;
}







   }  // namespace Minuit2

}  // namespace ROOT
