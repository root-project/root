// @(#)root/minuit2:$Id$
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
#include "Minuit2/MnPrint.h"

#include "./MPIProcess.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cassert>
#include <iomanip>

namespace ROOT {

namespace Minuit2 {

FunctionGradient Numerical2PGradientCalculator::operator()(const MinimumParameters &par) const
{
   // calculate gradient using Initial gradient calculator and from MinimumParameters object

   FunctionGradient gra = calculateInitialGradient(par, fTransformation, fFcn.ErrorDef());

   return (*this)(par, gra);
}

FunctionGradient Numerical2PGradientCalculator::
operator()(const MinimumParameters &par, const FunctionGradient &Gradient) const
{
   // calculate numerical gradient from MinimumParameters object
   // the algorithm takes correctly care when the gradient is approximately zero

   MnUserTransformation const &trafo = fTransformation;

   //    std::cout<<"########### Numerical2PDerivative"<<std::endl;
   //    std::cout<<"initial grd: "<<Gradient.Grad()<<std::endl;
   //    std::cout<<"position: "<<par.Vec()<<std::endl;
   MnPrint print("Numerical2PGradientCalculator");

   assert(par.IsValid());

   double fcnmin = par.Fval();
   //   std::cout<<"fval: "<<fcnmin<<std::endl;

   double eps2 = trafo.Precision().Eps2();
   double eps = trafo.Precision().Eps();

   //print.Trace("Assumed precision eps", eps, "eps2", eps2);

   double dfmin = 8. * eps2 * (std::fabs(fcnmin) + fFcn.Up());
   double vrysml = 8. * eps * eps;
   //   double vrysml = std::max(1.e-4, eps2);
   //    std::cout<<"dfmin= "<<dfmin<<std::endl;
   //    std::cout<<"vrysml= "<<vrysml<<std::endl;
   //    std::cout << " ncycle " << Ncycle() << std::endl;

   unsigned int n = (par.Vec()).size();
   unsigned int ncycle = fStrategy.GradientNCycles();
   //   MnAlgebraicVector vgrd(n), vgrd2(n), vgstp(n);
   MnAlgebraicVector grd = Gradient.Grad();
   MnAlgebraicVector g2 = Gradient.G2();
   MnAlgebraicVector gstep = Gradient.Gstep();

   print.Debug("Calculating gradient around function value", fcnmin, "\n\t at point", par.Vec());

#ifndef _OPENMP

   MPIProcess mpiproc(n, 0);

   // for serial execution this can be outside the loop
   MnAlgebraicVector x = par.Vec();
   MnFcnCaller mfcnCaller{fFcn};

   unsigned int startElementIndex = mpiproc.StartElementIndex();
   unsigned int endElementIndex = mpiproc.EndElementIndex();

   for (unsigned int i = startElementIndex; i < endElementIndex; i++) {

#else

   // parallelize this loop using OpenMP
//#define N_PARALLEL_PAR 5
#pragma omp parallel for if (fDoParallelOMP)
   //#pragma omp for schedule (static, N_PARALLEL_PAR)

   for (unsigned int i = 0; i < n; i++) {

#endif

#ifdef _OPENMP
      // create in loop since each thread will use its own copy
      MnAlgebraicVector x = par.Vec();
      MnFcnCaller mfcnCaller{fFcn};
#endif

      double xtf = x(i);
      double epspri = eps2 + std::fabs(grd(i) * eps2);
      double stepb4 = 0.;
      for (unsigned int j = 0; j < ncycle; j++) {
         double optstp = std::sqrt(dfmin / (std::fabs(g2(i)) + epspri));
         double step = std::max(optstp, std::fabs(0.1 * gstep(i)));
         //       std::cout<<"step: "<<step;
         if (trafo.Parameter(trafo.ExtOfInt(i)).HasLimits()) {
            if (step > 0.5)
               step = 0.5;
         }
         double stpmax = 10. * std::fabs(gstep(i));
         if (step > stpmax)
            step = stpmax;
         //       std::cout<<" "<<step;
         double stpmin = std::max(vrysml, 8. * std::fabs(eps2 * x(i)));
         if (step < stpmin)
            step = stpmin;
         //       std::cout<<" "<<step<<std::endl;
         //       std::cout<<"step: "<<step<<std::endl;
         if (std::fabs((step - stepb4) / step) < fStrategy.GradientStepTolerance()) {
            //    std::cout<<"(step-stepb4)/step"<<std::endl;
            //    std::cout<<"j= "<<j<<std::endl;
            //    std::cout<<"step= "<<step<<std::endl;
            break;
         }
         gstep(i) = step;
         stepb4 = step;

         x(i) = xtf + step;
         double fs1 = mfcnCaller(x);
         x(i) = xtf - step;
         double fs2 = mfcnCaller(x);
         x(i) = xtf;

         double grdb4 = grd(i);
         grd(i) = 0.5 * (fs1 - fs2) / step;
         g2(i) = (fs1 + fs2 - 2. * fcnmin) / step / step;

#ifdef _OPENMP
#pragma omp critical
#endif
         {
#ifdef _OPENMP
            // must create thread-local MnPrint instances when printing inside threads
            MnPrint printtl("Numerical2PGradientCalculator[OpenMP]");
#endif
            if (i == 0 && j == 0) {
#ifdef _OPENMP
               printtl.Trace([&](std::ostream &os) {
#else
               print.Trace([&](std::ostream &os) {
#endif
                  os << std::setw(10) << "parameter" << std::setw(6) << "cycle" << std::setw(15) << "x" << std::setw(15)
                     << "step" << std::setw(15) << "f1" << std::setw(15) << "f2" << std::setw(15) << "grd"
                     << std::setw(15) << "g2" << std::endl;
               });
            }
#ifdef _OPENMP
            printtl.Trace([&](std::ostream &os) {
#else
            print.Trace([&](std::ostream &os) {
#endif
               const int pr = os.precision(13);
               const int iext = trafo.ExtOfInt(i);
               os << std::setw(10) << trafo.Name(iext) << std::setw(5) << j << "  " << x(i) << " " << step << " "
                  << fs1 << " " << fs2 << " " << grd(i) << " " << g2(i) << std::endl;
               os.precision(pr);
            });
         }

         if (std::fabs(grdb4 - grd(i)) / (std::fabs(grd(i)) + dfmin / step) < fStrategy.GradientTolerance()) {
            //    std::cout<<"j= "<<j<<std::endl;
            //    std::cout<<"step= "<<step<<std::endl;
            //    std::cout<<"fs1, fs2: "<<fs1<<" "<<fs2<<std::endl;
            //    std::cout<<"fs1-fs2: "<<fs1-fs2<<std::endl;
            break;
         }
      }

      //     vgrd(i) = grd;
      //     vgrd2(i) = g2;
      //     vgstp(i) = gstep;
   }

#ifndef _OPENMP
   mpiproc.SyncVector(grd);
   mpiproc.SyncVector(g2);
   mpiproc.SyncVector(gstep);
#endif

   // print after parallel processing to avoid synchronization issues
   print.Debug([&](std::ostream &os) {
      const int pr = os.precision(13);
      os << std::endl;
      os << std::setw(14) << "Parameter" << std::setw(14) << "Gradient" << std::setw(14) << "g2 " << std::setw(14)
         << "step" << std::endl;
      for (int i = 0; i < int(n); i++) {
         const int iext = trafo.ExtOfInt(i);
         os << std::setw(14) << trafo.Name(iext) << " " << grd(i) << " " << g2(i) << " " << gstep(i) << std::endl;
      }
      os.precision(pr);
   });

   return FunctionGradient(grd, g2, gstep);
}

} // namespace Minuit2

} // namespace ROOT
