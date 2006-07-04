// @(#)root/minuit2:$Name:  $:$Id: NegativeG2LineSearch.cxx,v 1.1 2005/11/29 14:43:31 moneta Exp $
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
         
         //       std::cout << "negative G2 - iter " << iter << " param " << i << "  grad2 " << dgrad.G2()(i) << " grad " << dgrad.Vec()(i) 
         // 		<< " grad step " << dgrad.Gstep()(i) << std::endl; 
         if(dgrad.G2()(i) <= 0) {      
            //       if(dgrad.G2()(i) < prec.Eps()) {
            // do line search if second derivative negative
            MnAlgebraicVector step(n);
            MnLineSearch lsearch;
            step(i) = dgrad.Gstep()(i)*dgrad.Vec()(i);
            //	if(fabs(dgrad.Vec()(i)) >  prec.Eps2()) 
            if(fabs(dgrad.Vec()(i)) >  0 ) 
               step(i) *= (-1./fabs(dgrad.Vec()(i)));
            double gdel = step(i)*dgrad.Vec()(i);
            MnParabolaPoint pp = lsearch(fcn, pa, step, gdel, prec);
            //	std::cout << " line search result " << pp.x() << "  " << pp.y() << std::endl;
            step *= pp.x();
            pa = MinimumParameters(pa.Vec() + step, pp.y());    
            dgrad = gc(pa, dgrad);         
            //  	std::cout << "Line search - iter" << iter << " param " << i << " step " << step(i) << " new grad2 " << dgrad.G2()(i) << " new grad " <<  dgrad.Vec()(i) << std::endl;
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
      //     if(grad.G2()(i) < prec.Eps2()) { 
      if(grad.G2()(i) <= 0 ) { 
         //      std::cout << "negative G2 " << i << "  grad " << grad.G2()(i) << " precision " << prec.Eps2() << std::endl;
         return true;
      }
         
   return false;
}

   }  // namespace Minuit2

}  // namespace ROOT
