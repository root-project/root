// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei, E.G.P. Bos   2003-2017

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 * Copyright (c) 2017 Patrick Bos, Netherlands eScience Center        *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnSeedGenerator.h"
#include "Minuit2/MinimumSeed.h"
#include "Minuit2/MnFcn.h"
#include "Minuit2/GradientCalculator.h"
#include "Minuit2/InitialGradientCalculator.h"
#include "Minuit2/MnUserTransformation.h"
#include "Minuit2/MinimumParameters.h"
#include "Minuit2/FunctionGradient.h"
#include "Minuit2/MinimumError.h"
#include "Minuit2/MnMatrix.h"
#include "Minuit2/MnMachinePrecision.h"
#include "Minuit2/MinuitParameter.h"
#include "Minuit2/MnLineSearch.h"
#include "Minuit2/MnParabolaPoint.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/VariableMetricEDMEstimator.h"
#include "Minuit2/NegativeG2LineSearch.h"
#include "Minuit2/AnalyticalGradientCalculator.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/HessianGradientCalculator.h"

//#define DEBUG

//#if defined(DEBUG) || defined(WARNINGMSG)
#include "Minuit2/MnPrint.h"
//#endif



#include <math.h>


namespace ROOT {

   namespace Minuit2 {


MinimumSeed MnSeedGenerator::operator()(const MnFcn& fcn, const GradientCalculator& gc, const MnUserParameterState& st, const MnStrategy& stra) const {

//  std::cout << "MnSeedGenerator::operator() for general GradientCalculator" << std::endl;

   // find seed (initial minimization point) using the calculated gradient
   unsigned int n = st.VariableParameters();
   const MnMachinePrecision& prec = st.Precision();

#ifdef DEBUG
   std::cout << "MnSeedGenerator: operator() - var par = " << n << " mnfcn pointer " << &fcn << std::endl;
#endif

   int printLevel = MnPrint::Level();

   // initial starting values
   MnAlgebraicVector x(n);
   for(unsigned int i = 0; i < n; i++) x(i) = st.IntParameters()[i];

//  std::cout << "-- MnSeedGenerator::operator(.., GradientCalculator, ..):" << std::endl;
//  for (int i = 0; i < n; ++i) {
//    std::cout << std::hexfloat << "x=("<< x(i) << ",\t";
//  }
//  std::cout << ")" << std::endl;

  double fcnmin = fcn(x);

   if (printLevel > 1) {
      std::cout << "MnSeedGenerator: for initial parameters FCN = ";
      MnPrint::PrintFcn(std::cout,fcnmin);
   }

   MinimumParameters pa(x, fcnmin);
//  std::cout << "... doing gc(pa) ..." << std::endl;
//  std::cout << "-- hier? 1 --" << std::endl;
   FunctionGradient dgrad = gc(pa);
//  std::cout << "-- hier? 2 --" << std::endl;
//  std::cout << "dgrad.Vec: " << dgrad.Vec() << std::endl;
//  std::cout << "dgrad.G2: " << dgrad.G2() << std::endl;
   MnAlgebraicSymMatrix mat(n);
   double dcovar = 1.;
   if(st.HasCovariance()) {
//     std::cout << "has covariance" << std::endl;
      for(unsigned int i = 0; i < n; i++)
         for(unsigned int j = i; j < n; j++) mat(i,j) = st.IntCovariance()(i,j);
      dcovar = 0.;
   } else {
//     std::cout << "has no covariance" << std::endl;
      for(unsigned int i = 0; i < n; i++)
         mat(i,i) = (fabs(dgrad.G2()(i)) > prec.Eps2() ? 1./dgrad.G2()(i) : 1.);
   }
   MinimumError err(mat, dcovar);

   double edm = VariableMetricEDMEstimator().Estimate(dgrad, err);
   MinimumState state(pa, err, dgrad, edm, fcn.NumOfCalls());

   if (printLevel >1) {
      MnPrint::PrintState(std::cout, state, "MnSeedGenerator: Initial state:  ");
   }

   NegativeG2LineSearch ng2ls;
//  std::cout << dgrad.Vec() << std::endl;
//  std::cout << prec << std::endl;
//  std::cout << ng2ls.HasNegativeG2(dgrad, prec) << std::endl;
//  std::cout << dgrad.Vec().size() << std::endl;

  if(ng2ls.HasNegativeG2(dgrad, prec)) {
#ifdef DEBUG
      std::cout << "MnSeedGenerator: Negative G2 Found: " << std::endl;
      std::cout << x << std::endl;
      std::cout << dgrad.Grad() << std::endl;
      std::cout << dgrad.G2() << std::endl;
#endif
      state = ng2ls(fcn, state, gc, prec);

      if (printLevel >1) {
         MnPrint::PrintState(std::cout, state, "MnSeedGenerator: Negative G2 found - new state:  ");
      }

   }


   if(stra.Strategy() == 2 && !st.HasCovariance()) {
      //calculate full 2nd derivative
#ifdef DEBUG
      std::cout << "MnSeedGenerator: calling MnHesse  " << std::endl;
#endif
      MinimumState tmp = MnHesse(stra)(fcn, state, st.Trafo());

      if (printLevel >1) {
         MnPrint::PrintState(std::cout, tmp, "MnSeedGenerator: run Hesse - new state:  ");
      }

//     std::cout << "-- MnSeedGenerator::operator(.., GradientCalculator, ..), strategy 2 end state:" << std::endl;
//     for (int i = 0; i < n; ++i) {
//       std::cout << std::hexfloat << "x=("<< tmp.Vec()(i) << ",\t";
//     }
//     std::cout << ")" << std::endl;

     return MinimumSeed(tmp, st.Trafo());
   }

//  std::cout << "-- MnSeedGenerator::operator(.., GradientCalculator, ..), regular end state:" << std::endl;
//  for (int i = 0; i < n; ++i) {
//    std::cout << std::hexfloat << "x=("<< state.Vec()(i) << ",\t";
//  }
//  std::cout << ")" << std::endl;

  return MinimumSeed(state, st.Trafo());
}


MinimumSeed MnSeedGenerator::operator()(const MnFcn& fcn, const AnalyticalGradientCalculator& gc, const MnUserParameterState& st, const MnStrategy& stra) const {

//  std::cout << "MnSeedGenerator::operator() for AnalyticalGradientCalculator" << std::endl;

  // find seed (initial point for minimization) using analytical gradient
   unsigned int n = st.VariableParameters();
   const MnMachinePrecision& prec = st.Precision();

   int printLevel = MnPrint::Level();

  // initial starting values
   MnAlgebraicVector x(n);
   for(unsigned int i = 0; i < n; i++) x(i) = st.IntParameters()[i];

//  std::cout << "-- MnSeedGenerator::operator(.., AnalyticalGradientCalculator, ..):" << std::endl;
//  for (int i = 0; i < n; ++i) {
//    std::cout << std::hexfloat << "x=("<< x(i) << ",\t";
//  }
//  std::cout << ")" << std::endl;

  double fcnmin = fcn(x);
   MinimumParameters pa(x, fcnmin);

//  std::cout << "... creating igc ..." << std::endl;
//   InitialGradientCalculator igc(fcn, st.Trafo(), stra);
//  std::cout << "... doing igc(pa) ..." << std::endl;
//   FunctionGradient tmp = igc(pa);
//  std::cout << "tmp.G2: " << tmp.G2() << std::endl;
//  std::cout << "-- hier? 1 --" << std::endl;
   FunctionGradient grd = gc(pa);
//  std::cout << "grd.G2: " << grd.G2() << std::endl;
//  std::cout << "-- hier? 2 --" << std::endl;

//  FunctionGradient dgrad(grd.Grad(), tmp.G2(), tmp.Gstep());
    FunctionGradient dgrad(grd.Grad(), grd.G2(), grd.Gstep());
//  std::cout << "dgrad.Vec: " << dgrad.Vec() << std::endl;
//  std::cout << "dgrad.G2: " << dgrad.G2() << std::endl;

   if(gc.CheckGradient()) {
      bool good = true;
      HessianGradientCalculator hgc(fcn, st.Trafo(), MnStrategy(2));
      std::pair<FunctionGradient, MnAlgebraicVector> hgrd = hgc.DeltaGradient(pa, dgrad);
      for(unsigned int i = 0; i < n; i++) {
         if(fabs(hgrd.first.Grad()(i) - grd.Grad()(i)) > hgrd.second(i)) {
#ifdef WARNINGMSG
            MN_INFO_MSG("MnSeedGenerator:gradient discrepancy of external Parameter too large");
            int externalParameterIndex = st.Trafo().ExtOfInt(i);
            const char * parameter_name = st.Trafo().Name(externalParameterIndex);
            MN_INFO_VAL(parameter_name);
            MN_INFO_VAL(externalParameterIndex);
            MN_INFO_VAL2("internal",i);
#endif
            good = false;
         }
      }
      if(!good) {
#ifdef WARNINGMSG
         MN_ERROR_MSG("Minuit does not accept user specified Gradient. To force acceptance, override 'virtual bool CheckGradient() const' of FCNGradientBase.h in the derived class.");
#endif
         assert(good);
      }
   }

   MnAlgebraicSymMatrix mat(n);
   double dcovar = 1.;
   if(st.HasCovariance()) {
//     std::cout << "has covariance" << std::endl;
      for(unsigned int i = 0; i < n; i++)
         for(unsigned int j = i; j < n; j++) mat(i,j) = st.IntCovariance()(i,j);
      dcovar = 0.;
   } else {
//     std::cout << "has no covariance" << std::endl;
      for(unsigned int i = 0; i < n; i++)
         mat(i,i) = (fabs(dgrad.G2()(i)) > prec.Eps2() ? 1./dgrad.G2()(i) : 1.);
   }
   MinimumError err(mat, dcovar);
   double edm = VariableMetricEDMEstimator().Estimate(dgrad, err);
   MinimumState state(pa, err, dgrad, edm, fcn.NumOfCalls());

   NegativeG2LineSearch ng2ls;
//  std::cout << dgrad.Vec() << std::endl;
//  std::cout << prec << std::endl;
//  std::cout << ng2ls.HasNegativeG2(dgrad, prec) << std::endl;
//  std::cout << dgrad.Vec().size() << std::endl;

   if(ng2ls.HasNegativeG2(dgrad, prec)) {
//      Numerical2PGradientCalculator ngc(fcn, st.Trafo(), stra);
//      state = ng2ls(fcn, state, ngc, prec);
      state = ng2ls(fcn, state, gc, prec);

     if (printLevel >1) {
       MnPrint::PrintState(std::cout, state, "MnSeedGenerator: Negative G2 found - new state:  ");
     }

   }

   if(stra.Strategy() == 2 && !st.HasCovariance()) {
      //calculate full 2nd derivative
      MinimumState tmpState = MnHesse(stra)(fcn, state, st.Trafo());

//     std::cout << "-- MnSeedGenerator::operator(.., AnalyticalGradientCalculator, ..), strategy 2 end state:" << std::endl;
//     for (int i = 0; i < n; ++i) {
//       std::cout << std::hexfloat << "x=("<< tmpState.Vec()(i) << ",\t";
//     }
//     std::cout << ")" << std::endl;

     return MinimumSeed(tmpState, st.Trafo());
   }

//  std::cout << "-- MnSeedGenerator::operator(.., AnalyticalGradientCalculator, ..), regular end state:" << std::endl;
//  for (int i = 0; i < n; ++i) {
//    std::cout << std::hexfloat << "x=("<< state.Vec()(i) << ",\t";
//  }
//  std::cout << ")" << std::endl;

  return MinimumSeed(state, st.Trafo());
}

   }  // namespace Minuit2

}  // namespace ROOT
