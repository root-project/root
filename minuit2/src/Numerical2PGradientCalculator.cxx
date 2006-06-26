// @(#)root/minuit2:$Name:  $:$Id: Numerical2PGradientCalculator.cxx,v 1.2 2006/06/26 07:56:43 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/InitialGradientCalculator.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MnStrategy.h"
#ifdef DEBUG
#include "Minuit2/MnPrint.h"
#endif

#include <math.h>

namespace ROOT {

   namespace Minuit2 {


FunctionGradient Numerical2PGradientCalculator::operator()(const MinimumParameters& par) const {
   // calculate gradient using Initial gradient calculator and from MinimumParameters object

   InitialGradientCalculator gc(fFcn, fTransformation, fStrategy);
   FunctionGradient gra = gc(par);
   
   return (*this)(par, gra);  
}


// comment it, because it was added
FunctionGradient Numerical2PGradientCalculator::operator()(const std::vector<double>& params) const {
   // calculate gradient from an std;:vector of paramteters
   
   int npar = params.size();
   
   MnAlgebraicVector par(npar);
   for (int i = 0; i < npar; ++i) {
      par(i) = params[i];
   }
   
   double fval = Fcn()(par);
   
   MinimumParameters minpars = MinimumParameters(par, fval);
   
   return (*this)(minpars);
   
}



FunctionGradient Numerical2PGradientCalculator::operator()(const MinimumParameters& par, const FunctionGradient& Gradient) const {
   // calculate numerical gradient from MinimumParameters object
   // the algorithm takes correctly care when the gradient is approximatly zero
   
   //    std::cout<<"########### Numerical2PDerivative"<<std::endl;
   //    std::cout<<"initial grd: "<<Gradient.Grad()<<std::endl;
   //    std::cout<<"position: "<<par.Vec()<<std::endl;
   
   assert(par.IsValid());
   
   MnAlgebraicVector x = par.Vec();
   
   double fcnmin = par.Fval();
   //   std::cout<<"fval: "<<fcnmin<<std::endl;
   
   double eps2 = Precision().Eps2(); 
   double eps = Precision().Eps();
   
   double dfmin = 8.*eps2*(fabs(fcnmin)+Fcn().Up());
   double vrysml = 8.*eps*eps;
   //   double vrysml = std::max(1.e-4, eps2);
   //    std::cout<<"dfmin= "<<dfmin<<std::endl;
   //    std::cout<<"vrysml= "<<vrysml<<std::endl;
   //    std::cout << " ncycle " << Ncycle() << std::endl;
   
   unsigned int n = x.size();
   unsigned int ncycle = Ncycle();
   //   MnAlgebraicVector vgrd(n), vgrd2(n), vgstp(n);
   MnAlgebraicVector grd = Gradient.Grad();
   MnAlgebraicVector g2 = Gradient.G2();
   MnAlgebraicVector gstep = Gradient.Gstep();
   for(unsigned int i = 0; i < n; i++) {
      double xtf = x(i);
      double epspri = eps2 + fabs(grd(i)*eps2);
      double stepb4 = 0.;
      for(unsigned int j = 0; j < ncycle; j++)  {
         double optstp = sqrt(dfmin/(fabs(g2(i))+epspri));
         double step = std::max(optstp, fabs(0.1*gstep(i)));
         //       std::cout<<"step: "<<step;
         if(Trafo().Parameter(Trafo().ExtOfInt(i)).HasLimits()) {
            if(step > 0.5) step = 0.5;
         }
         double stpmax = 10.*fabs(gstep(i));
         if(step > stpmax) step = stpmax;
         //       std::cout<<" "<<step;
         double stpmin = std::max(vrysml, 8.*fabs(eps2*x(i)));
         if(step < stpmin) step = stpmin;
         //       std::cout<<" "<<step<<std::endl;
         //       std::cout<<"step: "<<step<<std::endl;
         if(fabs((step-stepb4)/step) < StepTolerance()) {
            //  	std::cout<<"(step-stepb4)/step"<<std::endl;
            //  	std::cout<<"j= "<<j<<std::endl;
            //  	std::cout<<"step= "<<step<<std::endl;
            break;
         }
         gstep(i) = step;
         stepb4 = step;
         //       MnAlgebraicVector pstep(n);
         //       pstep(i) = step;
         //       double fs1 = Fcn()(pstate + pstep);
         //       double fs2 = Fcn()(pstate - pstep);
         
         x(i) = xtf + step;
         double fs1 = Fcn()(x);
         x(i) = xtf - step;
         double fs2 = Fcn()(x);
         x(i) = xtf;
         
         double grdb4 = grd(i);
         grd(i) = 0.5*(fs1 - fs2)/step;
         g2(i) = (fs1 + fs2 - 2.*fcnmin)/step/step;
         
         if(fabs(grdb4-grd(i))/(fabs(grd(i))+dfmin/step) < GradTolerance())  {
            //  	std::cout<<"j= "<<j<<std::endl;
            //  	std::cout<<"step= "<<step<<std::endl;
            //  	std::cout<<"fs1, fs2: "<<fs1<<" "<<fs2<<std::endl;
            //  	std::cout<<"fs1-fs2: "<<fs1-fs2<<std::endl;
            break;
         }
      }
      
      //     vgrd(i) = grd;
      //     vgrd2(i) = g2;
      //     vgstp(i) = gstep;
   }  
   //   std::cout<<"final grd: "<<grd<<std::endl;
   //   std::cout<<"########### return from Numerical2PDerivative"<<std::endl;
   return FunctionGradient(grd, g2, gstep);
}

const MnMachinePrecision& Numerical2PGradientCalculator::Precision() const {
   return fTransformation.Precision();
}

unsigned int Numerical2PGradientCalculator::Ncycle() const {
   return Strategy().GradientNCycles();
}

double Numerical2PGradientCalculator::StepTolerance() const {
   return Strategy().GradientStepTolerance();
}

double Numerical2PGradientCalculator::GradTolerance() const {
   return Strategy().GradientTolerance();
}


  }  // namespace Minuit2

}  // namespace ROOT
