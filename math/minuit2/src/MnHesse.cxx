// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MnHesse.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/MnPosDef.h"
#include "Minuit2/HessianGradientCalculator.h"
#include "Minuit2/Numerical2PGradientCalculator.h"
#include "Minuit2/InitialGradientCalculator.h"
#include "Minuit2/AnalyticalGradientCalculator.h"
#include "Minuit2/ExternalInternalGradientCalculator.h"
#include "Minuit2/MinimumState.h"
#include "Minuit2/VariableMetricEDMEstimator.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"

#include "./MPIProcess.h"
#include "Minuit2/MnFcn.h"

#include "Math/Util.h"

namespace ROOT {

namespace Minuit2 {

namespace {

/// Internal function to compute the Hessian using numerical derivative
/// computation.
MinimumState ComputeNumerical(const MnFcn &, const MinimumState &, const MnUserTransformation &, unsigned int maxcalls,
                              MnStrategy const &strat);

/// Internal function to compute the Hessian using an analytical computation or
/// externally provided in the FCNBase class.
MinimumState ComputeAnalytical(const FCNBase &, const MinimumState &, const MnUserTransformation &);

} // namespace

MnUserParameterState
MnHesse::operator()(const FCNBase &fcn, const MnUserParameterState &state, unsigned int maxcalls) const
{
   // interface from MnUserParameterState
   // create a new Minimum state and use that interface
   unsigned int n = state.VariableParameters();
   MnFcn mfcn{fcn, state.Trafo(), static_cast<int>(state.NFcn())};
   MnAlgebraicVector x(state.IntParameters());
   double amin = MnFcnCaller{mfcn}(x);
   MinimumParameters par(x, amin);
   // check if we can use analytical gradient
   if (fcn.HasGradient()) {
      // no need to compute gradient here
      MinimumState tmp = ComputeAnalytical(fcn, {par, {MnAlgebraicSymMatrix(n), 1.}, FunctionGradient(n),
        state.Edm(), static_cast<int>(state.NFcn())}, state.Trafo());
      return MnUserParameterState(tmp, fcn.Up(), state.Trafo());
   }
   // case of numerical gradient
   MinimumState tmp = ComputeNumerical(mfcn,
                                       {par,
                                        {MnAlgebraicSymMatrix(n), 1.},
                                        Numerical2PGradientCalculator{mfcn, state.Trafo(), fStrategy}(par),
                                        state.Edm(),
                                        static_cast<int>(state.NFcn())},
                                       state.Trafo(), maxcalls, fStrategy);
   return MnUserParameterState(tmp, fcn.Up(), state.Trafo());
}

void MnHesse::operator()(const FCNBase &fcn, FunctionMinimum &min, unsigned int maxcalls) const
{
   // interface from FunctionMinimum to be used after minimization
   // use last state from the minimization without the need to re-create a new state
   // do not reset function calls and keep updating them
   MnFcn mfcn{fcn, min.UserState().Trafo(), min.NFcn()};
   MinimumState st = (*this)(mfcn, min.State(), min.UserState().Trafo(), maxcalls);
   min.Add(st);
}

MinimumState MnHesse::operator()(const MnFcn &mfcn, const MinimumState &st, const MnUserTransformation &trafo,
                                 unsigned int maxcalls) const
{
   // check first if we have an analytical gradient
   if (st.Gradient().IsAnalytical()) {
      // check if we can compute analytical Hessian
      if (mfcn.Fcn().HasGradient() && mfcn.Fcn().HasHessian()) {
         return ComputeAnalytical(mfcn.Fcn(), st, trafo);
      }
   }
   // case of numerical computation or only analytical first derivatives
   return ComputeNumerical(mfcn, st, trafo, maxcalls, fStrategy);
}

namespace {

MinimumState ComputeAnalytical(const FCNBase &fcn, const MinimumState &st, const MnUserTransformation &trafo)
{
   unsigned int n = st.Parameters().Vec().size();
   MnAlgebraicSymMatrix vhmat(n);

   MnPrint print("MnHesse");
   ROOT::Math::Util::TimingScope timingScope([&print](std::string const& s){ print.Info(s); }, "Done after");

   const MnMachinePrecision &prec = trafo.Precision();

   std::unique_ptr<AnalyticalGradientCalculator> hc;
   if (fcn.gradParameterSpace() == GradientParameterSpace::Internal) {
      hc = std::make_unique<ExternalInternalGradientCalculator>(fcn,trafo);
   }  else {
      hc = std::make_unique<AnalyticalGradientCalculator>(fcn,trafo);
   }

   bool ret = hc->Hessian(st.Parameters(), vhmat);
   if (!ret) {
      print.Error("Error computing analytical Hessian. MnHesse fails and will return a null matrix");
      return MinimumState(st.Parameters(), MinimumError(vhmat, MinimumError::MnHesseFailed), st.Gradient(), st.Edm(),
                             st.NFcn());
   }
   MnAlgebraicVector g2(n);
   for (unsigned int i = 0; i < n; i++)
      g2(i) = vhmat(i,i);

   // update Function gradient with new G2 found
   FunctionGradient gr(st.Gradient().Grad(), g2);

   // verify if matrix pos-def (still 2nd derivative)
   print.Debug("Original error matrix", vhmat);

   MinimumError tmpErr = MnPosDef()(MinimumError(vhmat, 1.), prec);
   vhmat = tmpErr.InvHessian();

   print.Debug("PosDef error matrix", vhmat);

   int ifail = Invert(vhmat);
   if (ifail != 0) {

      print.Warn("Matrix inversion fails; will return diagonal matrix");

      MnAlgebraicSymMatrix tmpsym(vhmat.Nrow());
      for (unsigned int j = 0; j < n; j++) {
         tmpsym(j,j) = 1. / g2(j);
      }

      return MinimumState(st.Parameters(), MinimumError(tmpsym, MinimumError::MnInvertFailed), gr, st.Edm(), st.NFcn());
   }

   VariableMetricEDMEstimator estim;

   // if matrix is made pos def returns anyway edm
   if (tmpErr.IsMadePosDef()) {
      MinimumError err(vhmat, MinimumError::MnMadePosDef);
      double edm = estim.Estimate(gr, err);
      return MinimumState(st.Parameters(), err, gr, edm, st.NFcn());
   }

   // calculate edm for good errors
   MinimumError err(vhmat, 0.);
   // this use only grad values
   double edm = estim.Estimate(gr, err);

   print.Debug("Hessian is ACCURATE. New state:", "\n  First derivative:", st.Gradient().Grad(),
                "\n  Covariance matrix:", vhmat, "\n  Edm:", edm);

   return MinimumState(st.Parameters(), err, gr, edm, st.NFcn());
}

MinimumState ComputeNumerical(const MnFcn &mfcn, const MinimumState &st, const MnUserTransformation &trafo,
                              unsigned int maxcalls, MnStrategy const &strat)
{
   // internal interface from MinimumState and MnUserTransformation
   // Function who does the real Hessian calculations
   MnPrint print("MnHesse");

   ROOT::Math::Util::TimingScope timingScope([&print](std::string const& s){ print.Info(s); }, "Done after");

   MnFcnCaller mfcnCaller{mfcn};

   const MnMachinePrecision &prec = trafo.Precision();
   // make sure starting at the right place
   double amin = mfcnCaller(st.Vec());
   double aimsag = std::sqrt(prec.Eps2()) * (std::fabs(amin) + mfcn.Up());

   // diagonal Elements first

   unsigned int n = st.Parameters().Vec().size();
   if (maxcalls == 0)
      maxcalls = 200 + 100 * n + 5 * n * n;

   MnAlgebraicSymMatrix vhmat(n);
   MnAlgebraicVector g2 = st.Gradient().G2();
   MnAlgebraicVector gst = st.Gradient().Gstep();
   MnAlgebraicVector grd = st.Gradient().Grad();
   MnAlgebraicVector dirin = st.Gradient().Gstep();
   MnAlgebraicVector yy(n);

   // case gradient is not numeric (could be analytical or from FumiliGradientCalculator)

   if (st.Gradient().IsAnalytical()) {
      print.Info("Using analytical gradient but a numerical Hessian calculator - it could be not optimal");
      Numerical2PGradientCalculator igc(mfcn, trafo, strat);
      // should we check here if numerical gradient is compatible with analytical one ?
      FunctionGradient tmp = igc(st.Parameters());
      gst = tmp.Gstep();
      dirin = tmp.Gstep();
      g2 = tmp.G2();
      print.Warn("Analytical calculator ",grd," numerical ",tmp.Grad()," g2 ",g2);
   }

   MnAlgebraicVector x = st.Parameters().Vec();

   print.Debug("Gradient is", st.Gradient().IsAnalytical() ? "analytical" : "numerical", "\n  point:", x,
               "\n  fcn  :", amin, "\n  grad :", grd, "\n  step :", gst, "\n  g2   :", g2);

   for (unsigned int i = 0; i < n; i++) {

      double xtf = x(i);
      double dmin = 8. * prec.Eps2() * (std::fabs(xtf) + prec.Eps2());
      double d = std::fabs(gst(i));
      if (d < dmin)
         d = dmin;

      print.Debug("Derivative parameter", i, "d =", d, "dmin =", dmin);

      for (unsigned int icyc = 0; icyc < strat.HessianNCycles(); icyc++) {
         double sag = 0.;
         double fs1 = 0.;
         double fs2 = 0.;
         for (unsigned int multpy = 0; multpy < 5; multpy++) {
            x(i) = xtf + d;
            fs1 = mfcnCaller(x);
            x(i) = xtf - d;
            fs2 = mfcnCaller(x);
            x(i) = xtf;
            sag = 0.5 * (fs1 + fs2 - 2. * amin);

            print.Debug("cycle", icyc, "mul", multpy, "\tsag =", sag, "d =", d);

            //  Now as F77 Minuit - check that sag is not zero
            if (sag != 0)
               goto L30; // break
            if (trafo.Parameter(i).HasLimits()) {
               if (d > 0.5)
                  goto L26;
               d *= 10.;
               if (d > 0.5)
                  d = 0.51;
               continue;
            }
            d *= 10.;
         }

      L26:
         // get parameter name for i
         // (need separate scope for avoiding compl error when declaring name)
         print.Warn("2nd derivative zero for parameter", trafo.Name(trafo.ExtOfInt(i)),
                    "; MnHesse fails and will return diagonal matrix");

         for (unsigned int j = 0; j < n; j++) {
            double tmp = g2(j) < prec.Eps2() ? 1. : 1. / g2(j);
            vhmat(j, j) = tmp < prec.Eps2() ? 1. : tmp;
         }

         return MinimumState(st.Parameters(), MinimumError(vhmat, MinimumError::MnHesseFailed), st.Gradient(), st.Edm(),
                             mfcn.NumOfCalls());

      L30:
         double g2bfor = g2(i);
         g2(i) = 2. * sag / (d * d);
         grd(i) = (fs1 - fs2) / (2. * d);
         gst(i) = d;
         dirin(i) = d;
         yy(i) = fs1;
         double dlast = d;
         d = std::sqrt(2. * aimsag / std::fabs(g2(i)));
         if (trafo.Parameter(i).HasLimits())
            d = std::min(0.5, d);
         if (d < dmin)
            d = dmin;

         print.Debug("g1 =", grd(i), "g2 =", g2(i), "step =", gst(i), "d =", d, "diffd =", std::fabs(d - dlast) / d,
                     "diffg2 =", std::fabs(g2(i) - g2bfor) / g2(i));

         // see if converged
         if (std::fabs((d - dlast) / d) < strat.HessianStepTolerance())
            break;
         if (std::fabs((g2(i) - g2bfor) / g2(i)) < strat.HessianG2Tolerance())
            break;
         d = std::min(d, 10. * dlast);
         d = std::max(d, 0.1 * dlast);
      }
      vhmat(i, i) = g2(i);
      if (mfcn.NumOfCalls() > maxcalls) {

         // std::cout<<"maxcalls " << maxcalls << " " << mfcn.NumOfCalls() << "  " <<   st.NFcn() << std::endl;
         print.Warn("Maximum number of allowed function calls exhausted; will return diagonal matrix");

         for (unsigned int j = 0; j < n; j++) {
            double tmp = g2(j) < prec.Eps2() ? 1. : 1. / g2(j);
            vhmat(j, j) = tmp < prec.Eps2() ? 1. : tmp;
         }

         return MinimumState(st.Parameters(), MinimumError(vhmat, MinimumError::MnReachedCallLimit), st.Gradient(),
                             st.Edm(), mfcn.NumOfCalls());
      }
   }

   print.Debug("Second derivatives", g2);

   if (strat.Strategy() > 0) {
      // refine first derivative
      HessianGradientCalculator hgc(mfcn, trafo, strat);
      FunctionGradient gr = hgc(st.Parameters(), FunctionGradient(grd, g2, gst));
      // update gradient and step values
      grd = gr.Grad();
      gst = gr.Gstep();
   }

   // off-diagonal Elements
   // initial starting values
   bool doCentralFD = strat.HessianCentralFDMixedDerivatives();
   if (n > 0) {
      MPIProcess mpiprocOffDiagonal(n * (n - 1) / 2, 0);
      unsigned int startParIndexOffDiagonal = mpiprocOffDiagonal.StartElementIndex();
      unsigned int endParIndexOffDiagonal = mpiprocOffDiagonal.EndElementIndex();

      unsigned int offsetVect = 0;
      for (unsigned int in = 0; in < startParIndexOffDiagonal; in++)
         if ((in + offsetVect) % (n - 1) == 0)
            offsetVect += (in + offsetVect) / (n - 1);

      for (unsigned int in = startParIndexOffDiagonal; in < endParIndexOffDiagonal; in++) {

         int i = (in + offsetVect) / (n - 1);
         if ((in + offsetVect) % (n - 1) == 0)
            offsetVect += i;
         int j = (in + offsetVect) % (n - 1) + 1;

         if ((i + 1) == j || in == startParIndexOffDiagonal)
            x(i) += dirin(i);

         x(j) += dirin(j);

         double fs1 = mfcnCaller(x);
         if(!doCentralFD) {
            double elem = (fs1 + amin - yy(i) - yy(j)) / (dirin(i) * dirin(j));
            vhmat(i, j) = elem;
            x(j) -= dirin(j);
         } else {
            // three more function evaluations required for central fd
            x(i) -= dirin(i);
            x(i) -= dirin(i);
            double fs3 = mfcnCaller(x);
            x(j) -= dirin(j);
            x(j) -= dirin(j);
            double fs4 = mfcnCaller(x);
            x(i) += dirin(i);
            x(i) += dirin(i);
            double fs2 = mfcnCaller(x);
            x(j) += dirin(j);
            double elem = (fs1 - fs2 - fs3 + fs4)/(4.*dirin(i)*dirin(j));
            vhmat(i, j) = elem;
         }

         if (j % (n - 1) == 0 || in == endParIndexOffDiagonal - 1)
            x(i) -= dirin(i);
      }

      mpiprocOffDiagonal.SyncSymMatrixOffDiagonal(vhmat);
   }

   // verify if matrix pos-def (still 2nd derivative)
   // Note that for cases of extreme spread of eigenvalues, numerical precision
   // can mean the hessian is computed as being not pos-def
   // but the inverse of it is.

   print.Debug("Original error matrix", vhmat);

   MinimumError tmpErr = MnPosDef()(MinimumError(vhmat, 1.), prec); // pos-def version of hessian

   if (strat.HessianForcePosDef()) {
      vhmat = tmpErr.InvHessian();
   }

   print.Debug("PosDef error matrix", vhmat);

   int ifail = Invert(vhmat);
   if (ifail != 0) {

      print.Warn("Matrix inversion fails; will return diagonal matrix");

      MnAlgebraicSymMatrix tmpsym(vhmat.Nrow());
      for (unsigned int j = 0; j < n; j++) {
         double tmp = g2(j) < prec.Eps2() ? 1. : 1. / g2(j);
         tmpsym(j, j) = tmp < prec.Eps2() ? 1. : tmp;
      }

      return MinimumState(st.Parameters(), MinimumError(tmpsym, MinimumError::MnInvertFailed), st.Gradient(), st.Edm(),
                          mfcn.NumOfCalls());
   }

   FunctionGradient gr(grd, g2, gst);
   VariableMetricEDMEstimator estim;

   // if matrix is made pos def returns anyway edm
   if (tmpErr.IsMadePosDef()) {
      MinimumError err(vhmat, strat.HessianForcePosDef() ? MinimumError::MnMadePosDef : MinimumError::MnNotPosDef);
      double edm = estim.Estimate(gr, err);
      return MinimumState(st.Parameters(), err, gr, edm, mfcn.NumOfCalls());
   }

   // calculate edm for good errors
   MinimumError err(vhmat, 0.);
   double edm = estim.Estimate(gr, err);

   print.Debug("Hessian is ACCURATE. New state:", "\n  First derivative:", grd, "\n  Second derivative:", g2,
               "\n  Gradient step:", gst, "\n  Covariance matrix:", vhmat, "\n  Edm:", edm);

   return MinimumState(st.Parameters(), err, gr, edm, mfcn.NumOfCalls());
}

} // namespace

/*
 MinimumError MnHesse::Hessian(const MnFcn& mfcn, const MinimumState& st, const MnUserTransformation& trafo) const {

    const MnMachinePrecision& prec = trafo.Precision();
    // make sure starting at the right place
    double amin = mfcn(st.Vec());
    //   if(std::fabs(amin - st.Fval()) > prec.Eps2()) std::cout<<"function Value differs from amin  by "<<amin -
 st.Fval()<<std::endl;

    double aimsag = std::sqrt(prec.Eps2())*(std::fabs(amin)+mfcn.Up());

    // diagonal Elements first

    unsigned int n = st.Parameters().Vec().size();
    MnAlgebraicSymMatrix vhmat(n);
    MnAlgebraicVector g2 = st.Gradient().G2();
    MnAlgebraicVector gst = st.Gradient().Gstep();
    MnAlgebraicVector grd = st.Gradient().Grad();
    MnAlgebraicVector dirin = st.Gradient().Gstep();
    MnAlgebraicVector yy(n);
    MnAlgebraicVector x = st.Parameters().Vec();

    for(unsigned int i = 0; i < n; i++) {

       double xtf = x(i);
       double dmin = 8.*prec.Eps2()*std::fabs(xtf);
       double d = std::fabs(gst(i));
       if(d < dmin) d = dmin;
       for(int icyc = 0; icyc < Ncycles(); icyc++) {
          double sag = 0.;
          double fs1 = 0.;
          double fs2 = 0.;
          for(int multpy = 0; multpy < 5; multpy++) {
             x(i) = xtf + d;
             fs1 = mfcn(x);
             x(i) = xtf - d;
             fs2 = mfcn(x);
             x(i) = xtf;
             sag = 0.5*(fs1+fs2-2.*amin);
             if(sag > prec.Eps2()) break;
             if(trafo.Parameter(i).HasLimits()) {
                if(d > 0.5) {
                   std::cout<<"second derivative zero for Parameter "<<i<<std::endl;
                   std::cout<<"return diagonal matrix "<<std::endl;
                   for(unsigned int j = 0; j < n; j++) {
                      vhmat(j,j) = (g2(j) < prec.Eps2() ? 1. : 1./g2(j));
                      return MinimumError(vhmat, 1., false);
                   }
                }
                d *= 10.;
                if(d > 0.5) d = 0.51;
                continue;
             }
             d *= 10.;
          }
          if(sag < prec.Eps2()) {
             std::cout<<"MnHesse: internal loop exhausted, return diagonal matrix."<<std::endl;
             for(unsigned int i = 0; i < n; i++)
                vhmat(i,i) = (g2(i) < prec.Eps2() ? 1. : 1./g2(i));
             return MinimumError(vhmat, 1., false);
          }
          double g2bfor = g2(i);
          g2(i) = 2.*sag/(d*d);
          grd(i) = (fs1-fs2)/(2.*d);
          gst(i) = d;
          dirin(i) = d;
          yy(i) = fs1;
          double dlast = d;
          d = std::sqrt(2.*aimsag/std::fabs(g2(i)));
          if(trafo.Parameter(i).HasLimits()) d = std::min(0.5, d);
          if(d < dmin) d = dmin;

          // see if converged
          if(std::fabs((d-dlast)/d) < Tolerstp()) break;
          if(std::fabs((g2(i)-g2bfor)/g2(i)) < TolerG2()) break;
          d = std::min(d, 10.*dlast);
          d = std::max(d, 0.1*dlast);
       }
       vhmat(i,i) = g2(i);
    }

    //off-diagonal Elements
    for(unsigned int i = 0; i < n; i++) {
       x(i) += dirin(i);
       for(unsigned int j = i+1; j < n; j++) {
          x(j) += dirin(j);
          double fs1 = mfcn(x);
          double elem = (fs1 + amin - yy(i) - yy(j))/(dirin(i)*dirin(j));
          vhmat(i,j) = elem;
          x(j) -= dirin(j);
       }
       x(i) -= dirin(i);
    }

    return MinimumError(vhmat, 0.);
 }
 */

} // namespace Minuit2

} // namespace ROOT
