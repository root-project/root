#include "gtest/gtest.h"

#include "TF1.h"
#include "TGraph.h"
#include "TGraphErrors.h"

// https://github.com/root-project/root/issues/13895
//
// A pol1 fit of a TGraphErrors diverges when the x-errors dominate the y-errors
// and the true slope is negative. With coordinate errors HFitImpl switches the
// linear fitter off, so the polynomial goes to the minimizer; the seeding block
// that follows only covers gaus/expo/landau, so the fit starts from whatever
// parameters the function already carries. On the data below that produced
// p0 ~ -1.8e4 and p1 ~ +9.0e3 instead of p0 = 7 and p1 = -1 -- the slope came
// out with the wrong sign.
//
// Every case constructs its OWN TF1 instead of using the global "pol1": a pol1
// fit performed earlier in the same process leaves good parameters behind and
// silently masks the bug (fitting a plain TGraph first is enough to hide it).
//
// Tolerances: the data is exactly collinear, but with coordinate errors the fit
// is solved by the minimizer rather than exactly, so a few 1e-3 is the honest
// bound; that is still four orders of magnitude away from the divergence above.

namespace {

// y = 7 - x, with x-errors ten times the y-errors.
TGraphErrors makeNegativeSlopeGraph()
{
   const int n = 3;
   double x[n] = {1., 2., 3.};
   double y[n] = {6., 5., 4.};
   double ex[n] = {1., 1., 1.};
   double ey[n] = {0.1, 0.1, 0.1};
   return TGraphErrors(n, x, y, ex, ey);
}

} // namespace

TEST(TGraphErrorsPolFit, NegativeSlopeWithCoordinateErrors)
{
   TGraphErrors gr = makeNegativeSlopeGraph();
   TF1 pol("polFitNeg", "pol1", 0., 4.);

   ASSERT_EQ(0, gr.Fit(&pol, "Q"));

   EXPECT_NEAR(7., pol.GetParameter(0), 1e-2);
   EXPECT_NEAR(-1., pol.GetParameter(1), 1e-2);
}

// The positive-slope case already converged from the default parameters, so it
// guards against a seeding change breaking what worked.
TEST(TGraphErrorsPolFit, PositiveSlopeWithCoordinateErrors)
{
   const int n = 3;
   double x[n] = {1., 2., 3.};
   double y[n] = {4., 5., 6.}; // y = 3 + x
   double ex[n] = {1., 1., 1.};
   double ey[n] = {0.1, 0.1, 0.1};
   TGraphErrors gr(n, x, y, ex, ey);
   TF1 pol("polFitPos", "pol1", 0., 4.);

   ASSERT_EQ(0, gr.Fit(&pol, "Q"));

   EXPECT_NEAR(3., pol.GetParameter(0), 1e-2);
   EXPECT_NEAR(1., pol.GetParameter(1), 1e-2);
}

// Without coordinate errors the linear fitter is used and the result is exact;
// this pins that the path taken when there is nothing to seed is untouched.
TEST(TGraphErrorsPolFit, NegativeSlopeWithoutCoordinateErrors)
{
   const int n = 3;
   double x[n] = {1., 2., 3.};
   double y[n] = {6., 5., 4.};
   TGraph gr(n, x, y);
   TF1 pol("polFitNoErr", "pol1", 0., 4.);

   ASSERT_EQ(0, gr.Fit(&pol, "Q"));

   EXPECT_NEAR(7., pol.GetParameter(0), 1e-9);
   EXPECT_NEAR(-1., pol.GetParameter(1), 1e-9);
}

// The x-errors are what push the fit off the linear path, so ignoring them with
// the "EX0" option must reproduce the exact linear answer even on the graph that
// otherwise diverges.
TEST(TGraphErrorsPolFit, CoordinateErrorsIgnoredWithEX0)
{
   TGraphErrors gr = makeNegativeSlopeGraph();
   TF1 pol("polFitEX0", "pol1", 0., 4.);

   ASSERT_EQ(0, gr.Fit(&pol, "QEX0"));

   EXPECT_NEAR(7., pol.GetParameter(0), 1e-9);
   EXPECT_NEAR(-1., pol.GetParameter(1), 1e-9);
}
