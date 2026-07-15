#include <TMath.h>
#include <TF1.h>
#include <TGraph.h>
#include <TCanvas.h>

#include <gtest/gtest.h>

#include <array>

TEST(TMath, Gradient_Laplace)
{
   std::array<double, 5> parameters{2, 100., -1., -2., 0.1};
   std::array<double, 4> parameters1st{};
   std::array<double, 3> parameters2nd{};
   for (unsigned int i = 0; i < 4; ++i) {
      parameters1st[i] = parameters[i + 1] * (i + 1);
   }
   for (unsigned int i = 0; i < 3; ++i) {
      parameters2nd[i] = parameters1st[i + 1] * (i + 1);
   }

   TF1 poly("poly", "pol4", -10, 10);
   TF1 poly1st("derivative", "pol3", -10, 10);
   TF1 poly2nd("secondDerivative", "pol2", -10, 10);
   poly.SetParameters(parameters.data());
   poly1st.SetParameters(parameters1st.data());
   poly2nd.SetParameters(parameters2nd.data());

   constexpr std::size_t nPoint = 10000;
   std::array<double, nPoint> vx;
   std::array<double, nPoint> vPoly;
   for (unsigned int i = 0; i < nPoint; ++i) {
      const auto x = -10. + 20. / nPoint * i;
      vx[i] = x;
      vPoly[i] = poly.Eval(x);
   }

   auto grad = TMath::Gradient(nPoint, vPoly.data(), 20. / nPoint);
   auto lap = TMath::Laplacian(nPoint, vPoly.data(), 20. / nPoint);

   auto relativeDiff = [](double val, double ref) {
      if (ref == 0.) {
         return std::fabs(val - ref);
      }
      return std::fabs(val - ref) / ref;
   };

   // Check forward/backward differences
   EXPECT_LT(relativeDiff(grad[0], poly1st.Eval(-10)), 0.001);
   EXPECT_LT(relativeDiff(grad[nPoint - 1], poly1st.Eval(vx[nPoint - 1])), 0.001);
   EXPECT_LT(relativeDiff(lap[0], poly2nd.Eval(-10)), 0.001);
   EXPECT_LT(relativeDiff(lap[nPoint - 1], poly2nd.Eval(vx[nPoint - 1])), 0.001);

   {
      double squaredDiff_grad = 0.;
      double maxRelDiff_grad = 0.;
      double squaredDiff_laplace = 0.;
      double maxRelDiff_laplace = 0.;
      // The points on the edges are forward/backward differences, so they will diverge more
      // Therefore, run the comparison only on the centre
      for (unsigned int i = 1; i < nPoint - 1; ++i) {
         const auto x = vx[i];
         const double diff = poly1st.Eval(x) - grad[i];
         squaredDiff_grad += diff * diff;
         maxRelDiff_grad = std::max(relativeDiff(grad[i], poly1st.Eval(x)), maxRelDiff_grad);

         const double diff2 = poly2nd.Eval(x) - lap[i];
         squaredDiff_laplace += diff2 * diff2;
         maxRelDiff_laplace = std::max(relativeDiff(lap[i], poly2nd.Eval(x)), maxRelDiff_laplace);
      }

      // Central differences
      EXPECT_LT(maxRelDiff_grad, 0.01);
      EXPECT_LT(std::sqrt(squaredDiff_grad), 0.01);

      EXPECT_LT(maxRelDiff_laplace, 0.01);
      EXPECT_LT(std::sqrt(squaredDiff_laplace), 0.01);
   }

   constexpr bool plot = false;
   if constexpr (plot) {
      std::array<double, nPoint> vFirstDer;
      std::array<double, nPoint> vSecondDer;
      for (unsigned int i = 0; i < nPoint; ++i) {
         const auto x = -10. + 20. / nPoint * i;
         vFirstDer[i] = poly1st.Eval(x);
         vSecondDer[i] = poly2nd.Eval(x);
      }

      poly1st.SetLineColor(kBlue);
      poly2nd.SetLineColor(kGreen);

      TGraph g_grad(nPoint, vx.data(), grad);
      TGraph g_lap(nPoint, vx.data(), lap);

      g_grad.SetMarkerColor(kBlue);
      g_grad.SetMarkerStyle(5);
      g_lap.SetMarkerColor(kGreen);
      g_lap.SetMarkerStyle(5);

      TCanvas c;
      poly.Draw();
      poly1st.Draw("same");
      poly2nd.Draw("same");
      g_grad.DrawClone("P");
      g_lap.DrawClone("P");
      c.SaveAs("/tmp/MathCoreTests.png");
   }
}