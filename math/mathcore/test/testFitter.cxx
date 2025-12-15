// Tests for the ROOT::Fit::Fitter

#include <Fit/Fitter.h>
#include <Math/Functor.h>

#include <gtest/gtest.h>

namespace {

double gauss(double *x, double *p)
{
   double A = p[0];
   double mu = p[1];
   double sigma = p[2];
   double xx = x[0];
   return A * exp(-(xx - mu) * (xx - mu) / (2 * sigma * sigma));
}

} // namespace

/// Check if we can fix and release parameters, and that this will be correctly
/// reflected in the FitResult.
/// Covers https://github.com/root-project/root/issues/20703
TEST(Fitter, FitAndReleaseParams)
{
   // --- Create data ---
   const int n = 50;
   double x[n];
   double y[n];
   std::vector<double> initVals{1.0, 2.0, 1.0};
   for (int i = 0; i < n; ++i) {
      x[i] = i * 0.1;
      y[i] = gauss(&x[i], initVals.data());
   }

   // --- Model functor ---
   ROOT::Math::Functor f(
      [&](const double *p) {
         double chi2 = 0.0;
         for (int i = 0; i < n; ++i) {
            double yi = gauss(&x[i], (double *)p);
            double diff = y[i] - yi;
            chi2 += diff * diff;
         }
         return chi2;
      },
      3);

   ROOT::Fit::Fitter fitter;

   std::vector<bool> fixState{false, false, false};
   fitter.Config().SetParamsSettings(initVals.size(), initVals.data());

   fitter.Config().ParSettings(0).SetName("A");
   fitter.Config().ParSettings(1).SetName("mu");
   fitter.Config().ParSettings(2).SetName("sigma");

   // Repeatedly run fits using the SAME fitter
   for (int iter = 0; iter < 6; ++iter) {
      bool fix = (iter % 2 == 0);
      int iparam = (iter / 2) % initVals.size();

      fixState[iparam] = fix;
      if (fix) {
         fitter.Config().ParSettings(iparam).Fix();
      } else {
         fitter.Config().ParSettings(iparam).Release();
      }

      fitter.FitFCN(f, nullptr, 3);
      const ROOT::Fit::FitResult &res = fitter.Result();

      for (unsigned int i = 0; i < initVals.size(); ++i) {
         EXPECT_EQ(res.IsParameterFixed(i), fixState[i]);
      }
   }
}
