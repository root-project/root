// Tests for the ROOT::Fit::Fitter

#include <Fit/Fitter.h>
#include <Math/Functor.h>
#include <Math/MinimizerOptions.h>

#include <TGraphErrors.h>
#include <TGraphAsymmErrors.h>
#include <TF1.h>
#include <TFitResult.h>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

double gauss(double *x, double *p)
{
   double A = p[0];
   double mu = p[1];
   double sigma = p[2];
   double xx = x[0];
   return A * exp(-(xx - mu) * (xx - mu) / (2 * sigma * sigma));
}

#ifdef R__USE_IMT

// Build a deterministic dataset so the exec-policy tests are reproducible.
TGraphErrors makeGraphErrors(int n)
{
   std::vector<double> x(n), y(n), ex(n), ey(n);
   for (int i = 0; i < n; ++i) {
      double xi = 0.01 * i;
      x[i] = xi;
      y[i] = 3.0 + 2.0 * xi + 0.1 * std::sin(xi);
      ex[i] = 0.05;
      ey[i] = 0.2;
   }
   return TGraphErrors(n, x.data(), y.data(), ex.data(), ey.data());
}

TGraphAsymmErrors makeGraphAsymmErrors(int n)
{
   std::vector<double> x(n), y(n), exl(n), exh(n), eyl(n), eyh(n);
   for (int i = 0; i < n; ++i) {
      double xi = 0.01 * i;
      x[i] = xi;
      y[i] = 3.0 + 2.0 * xi + 0.1 * std::sin(xi);
      exl[i] = 0.04;
      exh[i] = 0.06;
      eyl[i] = 0.15;
      eyh[i] = 0.25;
   }
   return TGraphAsymmErrors(n, x.data(), y.data(), exl.data(), exh.data(), eyl.data(), eyh.data());
}

// Fit a graph with errors on the x coordinate (which uses the "effective chi2",
// FitUtil::EvaluateChi2Effective) both sequentially and multi-threaded, and check
// the two execution policies converge to the same minimum.
template <class Graph>
void checkEffectiveChi2ExecPolicy(Graph &g)
{
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit2");

   TF1 fSeq("fSeq", "[0]+[1]*x", 0, 20);
   fSeq.SetParameters(1, 1);
   auto rSeq = g.Fit(&fSeq, "S Q N SERIAL");
   ASSERT_TRUE(rSeq.Get() != nullptr);
   ASSERT_TRUE(rSeq->IsValid());

   TF1 fMT("fMT", "[0]+[1]*x", 0, 20);
   fMT.SetParameters(1, 1);
   auto rMT = g.Fit(&fMT, "S Q N MULTITHREAD");
   ASSERT_TRUE(rMT.Get() != nullptr);
   ASSERT_TRUE(rMT->IsValid());

   // The tiny residual difference is only due to the order of the floating point
   // sum in the parallel reduction.
   EXPECT_NEAR(rSeq->Parameter(0), rMT->Parameter(0), 1e-6);
   EXPECT_NEAR(rSeq->Parameter(1), rMT->Parameter(1), 1e-6);
   EXPECT_NEAR(rSeq->MinFcnValue(), rMT->MinFcnValue(), 1e-4 * std::abs(rSeq->MinFcnValue()));
}

#endif // R__USE_IMT

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

#ifdef R__USE_IMT

/// Fitting a TGraphErrors with errors on the x coordinate (effective chi2) must
/// give the same result sequentially and multi-threaded.
/// Covers https://github.com/root-project/root/issues/10021
/// See also https://root-forum.cern.ch/t/multithreading-on-minuit/46225
TEST(Fitter, EffectiveChi2ExecPolicy)
{
   auto g = makeGraphErrors(2000);
   checkEffectiveChi2ExecPolicy(g);
}

/// Same as above but with asymmetric errors (also goes through the effective chi2 path).
TEST(Fitter, EffectiveChi2ExecPolicyAsymm)
{
   auto g = makeGraphAsymmErrors(1500);
   checkEffectiveChi2ExecPolicy(g);
}

#endif
