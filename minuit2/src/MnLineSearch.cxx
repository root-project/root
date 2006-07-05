// @(#)root/minuit2:$Name:  $:$Id: MnLineSearch.cxx,v 1.4 2006/07/04 10:36:52 moneta Exp $
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005  

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MnParabola.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/MnParabolaFactory.h"
#include "Minuit2/LaSum.h"

namespace ROOT {

   namespace Minuit2 {


/**  Perform a line search from position defined by the vector st
       along the direction step, where the length of vector step
       gives the expected position of Minimum.
       fcn is Value of function at the starting position ,  
       gdel (if non-zero) is df/dx along step at st. 
       Return a parabola point containing Minimum x posiiton and y (function Value)
*/

MnParabolaPoint MnLineSearch::operator()(const MnFcn& fcn, const MinimumParameters& st, const MnAlgebraicVector& step, double gdel, const MnMachinePrecision& prec) const {
   
   //*-*-*-*-*-*-*-*-*-*Perform a line search from position st along step   *-*-*-*-*-*-*-*
   //*-*                =========================================
   //*-* SLAMBG and ALPHA control the maximum individual steps allowed.
   //*-* The first step is always =1. The max length of second step is SLAMBG.
   //*-* The max size of subsequent steps is the maximum previous successful
   //*-*   step multiplied by ALPHA + the size of most recent successful step,
   //*-*   but cannot be smaller than SLAMBG.
   //*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
   
   //   std::cout<<"gdel= "<<gdel<<std::endl;
   //   std::cout<<"step= "<<step<<std::endl;
   
   double overal = 1000.;
   double undral = -100.;
   double toler = 0.05;
   double slamin = 0.;
   double slambg = 5.;
   double alpha = 2.;
   int maxiter = 12;
   int niter = 0;
   
   for(unsigned int i = 0; i < step.size(); i++) {
      if(step(i) == 0 )  continue;
      double ratio = fabs(st.Vec()(i)/step(i));
      if( slamin == 0) slamin = ratio;
      if(ratio < slamin) slamin = ratio;
   }
   if(fabs(slamin) < prec.Eps()) slamin = prec.Eps();
   slamin *= prec.Eps2();
   
   double f0 = st.Fval();
   double f1 = fcn(st.Vec()+step);
   double fvmin = st.Fval();
   double xvmin = 0.;
   
   if(f1 < f0) {
      fvmin = f1;
      xvmin = 1.;
   }
   double toler8 = toler;
   double slamax = slambg;
   double flast = f1;
   double slam = 1.;
   
   bool iterate = false;
   MnParabolaPoint p0(0., f0);
   MnParabolaPoint p1(slam, flast);
   double f2 = 0.;
   // quadratic interpolation using the two points p0,p1 and the slope at p0
   do {  
      // cut toler8 as function goes up 
      iterate = false;
      MnParabola pb = MnParabolaFactory()(p0, gdel, p1);
      //     std::cout<<"pb.Min() = "<<pb.Min()<<std::endl;
      //     std::cout<<"flast, f0= "<<flast<<", "<<f0<<std::endl;
      //     std::cout<<"flast-f0= "<<flast-f0<<std::endl;
      //     std::cout<<"slam= "<<slam<<std::endl;
      //     double df = flast-f0;
      //     if(fabs(df) < prec.Eps2()) {
      //       if(flast-f0 < 0.) df = -prec.Eps2();
      //       else df = prec.Eps2();
      //     }
      //     std::cout<<"df= "<<df<<std::endl;
      //     double denom = 2.*(df-gdel*slam)/(slam*slam);
      double denom = 2.*(flast-f0-gdel*slam)/(slam*slam);
      //     std::cout<<"denom= "<<denom<<std::endl;
      if(denom != 0) {
         slam =  - gdel/denom;
      } else {
         denom = -0.1*gdel;
         slam = 1.;
      } 
      //     std::cout<<"slam= "<<slam<<std::endl;
      if(slam < 0.) slam = slamax;
      //     std::cout<<"slam= "<<slam<<std::endl;
      if(slam > slamax) slam = slamax;
      //     std::cout<<"slam= "<<slam<<std::endl;
      if(slam < toler8) slam = toler8;
      //     std::cout<<"slam= "<<slam<<std::endl;
      if(slam < slamin) {
         //       std::cout<<"f1, f2= "<<p0.y()<<", "<<p1.y()<<std::endl;
         //       std::cout<<"x1, x2= "<<p0.x()<<", "<<p1.x()<<std::endl;
         //       std::cout<<"x, f= "<<xvmin<<", "<<fvmin<<std::endl;
         return MnParabolaPoint(xvmin, fvmin);
      }
      if(fabs(slam - 1.) < toler8 && p1.y() < p0.y()) {
         //       std::cout<<"f1, f2= "<<p0.y()<<", "<<p1.y()<<std::endl;
         //       std::cout<<"x1, x2= "<<p0.x()<<", "<<p1.x()<<std::endl;
         //       std::cout<<"x, f= "<<xvmin<<", "<<fvmin<<std::endl;
         return MnParabolaPoint(xvmin, fvmin);
      }
      if(fabs(slam - 1.) < toler8) slam = 1. + toler8;
      
      //     if(fabs(gdel) < prec.Eps2() && fabs(denom) < prec.Eps2())
      //       slam = 1000.;
      //     MnAlgebraicVector tmp = step;
      //     tmp *= slam;
      //     f2 = fcn(st.Vec()+tmp);
      f2 = fcn(st.Vec() + slam*step);
      if(f2 < fvmin) {
         fvmin = f2;
         xvmin = slam;
      }
      // LM : correct a bug using precision
      if (fabs( p0.y() - fvmin) < fabs(fvmin)*prec.Eps() ) { 
         //   if(p0.y()-prec.Eps() < fvmin && fvmin < p0.y()+prec.Eps()) {
         iterate = true;
         flast = f2;
         toler8 = toler*slam;
         overal = slam - toler8;
         slamax = overal;
         p1 = MnParabolaPoint(slam, flast);
         niter++;
         }
      } while(iterate && niter < maxiter);
   if(niter >= maxiter) {
      // exhausted max number of iterations
      return MnParabolaPoint(xvmin, fvmin);  
   }
   
   //   std::cout<<"after initial 2-point iter: "<<std::endl;
   //   std::cout<<"f0, f1, f2= "<<p0.y()<<", "<<p1.y()<<", "<<f2<<std::endl;
   //   std::cout<<"x0, x1, x2= "<<p0.x()<<", "<<p1.x()<<", "<<slam<<std::endl;
   
   MnParabolaPoint p2(slam, f2);
   
   // do now the quadratic interpolation with 3 points
   do {
      slamax = std::max(slamax, alpha*fabs(xvmin));
      MnParabola pb = MnParabolaFactory()(p0, p1, p2);
      //     std::cout<<"p2-p0,p1: "<<p2.y() - p0.y()<<", "<<p2.y() - p1.y()<<std::endl;
      //     std::cout<<"a, b, c= "<<pb.a()<<" "<<pb.b()<<" "<<pb.c()<<std::endl;
      if(pb.a() < prec.Eps2()) {
         double slopem = 2.*pb.a()*xvmin + pb.b();
         if(slopem < 0.) slam = xvmin + slamax;
         else slam = xvmin - slamax;
      } else {
         slam = pb.Min();
         //      std::cout<<"pb.Min() slam= "<<slam<<std::endl;
         if(slam > xvmin + slamax) slam = xvmin + slamax;
         if(slam < xvmin - slamax) slam = xvmin - slamax;
      }
      if(slam > 0.) {
         if(slam > overal) slam = overal;
      } else {
         if(slam < undral) slam = undral;
      }
      //     std::cout<<" slam= "<<slam<<std::endl;
      
      double f3 = 0.;
      do {
         iterate = false;
         double toler9 = std::max(toler8, fabs(toler8*slam));
         // min. of parabola at one point    
         if(fabs(p0.x() - slam) < toler9 || 
            fabs(p1.x() - slam) < toler9 || 
            fabs(p2.x() - slam) < toler9) {
            //   	std::cout<<"f1, f2, f3= "<<p0.y()<<", "<<p1.y()<<", "<<p2.y()<<std::endl;
            //   	std::cout<<"x1, x2, x3= "<<p0.x()<<", "<<p1.x()<<", "<<p2.x()<<std::endl;
            //   	std::cout<<"x, f= "<<xvmin<<", "<<fvmin<<std::endl;
            return MnParabolaPoint(xvmin, fvmin);
         }
         
         // take the step
         //       MnAlgebraicVector tmp = step;
         //       tmp *= slam;
         f3 = fcn(st.Vec() + slam*step);
         //       std::cout<<"f3= "<<f3<<std::endl;
         //       std::cout<<"f3-p(2-0).y()= "<<f3-p2.y()<<" "<<f3-p1.y()<<" "<<f3-p0.y()<<std::endl;
         // if latest point worse than all three previous, cut step
         if(f3 > p0.y() && f3 > p1.y() && f3 > p2.y()) {
            //   	std::cout<<"f3 worse than all three previous"<<std::endl;
            if(slam > xvmin) overal = std::min(overal, slam-toler8);
            if(slam < xvmin) undral = std::max(undral, slam+toler8);	
            slam = 0.5*(slam + xvmin);
            //   	std::cout<<"new slam= "<<slam<<std::endl;
            iterate = true;
            niter++;
         }
      } while(iterate && niter < maxiter);
      if(niter >= maxiter) {
         // exhausted max number of iterations
         return MnParabolaPoint(xvmin, fvmin);  
      }
      
      // find worst previous point out of three and replace
      MnParabolaPoint p3(slam, f3);
      if(p0.y() > p1.y() && p0.y() > p2.y()) p0 = p3;
      else if(p1.y() > p0.y() && p1.y() > p2.y()) p1 = p3;
      else p2 = p3;
      if(f3 < fvmin) {
         fvmin = f3;
         xvmin = slam;
      } else {
         if(slam > xvmin) overal = std::min(overal, slam-toler8);
         if(slam < xvmin) undral = std::max(undral, slam+toler8);	
      }
      
      niter++;
   } while(niter < maxiter);
   
   //   std::cout<<"f1, f2= "<<p0.y()<<", "<<p1.y()<<std::endl;
   //   std::cout<<"x1, x2= "<<p0.x()<<", "<<p1.x()<<std::endl;
   //   std::cout<<"x, f= "<<xvmin<<", "<<fvmin<<std::endl;
   return MnParabolaPoint(xvmin, fvmin);
   }
      
}  // namespace Minuit2

}  // namespace ROOT
