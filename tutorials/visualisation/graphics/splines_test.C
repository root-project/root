/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// \preview Examples of use of the spline classes.
///
/// \macro_image
/// \macro_code
///
/// \author Federico Carminati

void splines_test(Int_t np = 23, Double_t a = -0.5, Double_t b = 31)
{
   const Double_t power = 0.75;
   const Double_t eps = (b - a) * 1.e-5;

   // Define the original function
   TF1 *f = new TF1("f", "sin(x)*sin(x/10)", a - 0.05 * (b - a), b + 0.05 * (b - a));
   // Draw function
   f->SetLineColor(kBlue);
   f->Draw("lc");

   TSpline3 *spline3 = nullptr;
   TSpline5 *spline5 = nullptr;
   TLegend *legend = nullptr;

   for (Int_t nnp = 2; nnp <= np; ++nnp) {

      std::vector<Double_t> xx(nnp), yy(nnp);

      // Calculate the knots
      for (Int_t i = 0; i < nnp; ++i) {
         xx[i] = a + (b - a) * TMath::Power(i / Double_t(nnp - 1), power);
         yy[i] = f->Eval(xx[i]);
      }

      // Evaluate fifth spline coefficients
      delete spline5;
      spline5 = new TSpline5(TString::Format("quintic spline %dknt", nnp),
                             xx.data(), f, nnp, "b1e1b2e2", f->Derivative(a), f->Derivative(b),
                             (f->Derivative(a + eps) - f->Derivative(a)) / eps,
                             (f->Derivative(b) - f->Derivative(b - eps)) / eps);

      spline5->SetLineColor(kRed);
      spline5->SetLineWidth(3);

      // Draw the quintic spline
      spline5->Draw("lcsame");

      // Evaluate third spline coefficients
      delete spline3;
      spline3 = new TSpline3(TString::Format("third spline %dknt", nnp),
                           xx.data(), yy.data(), nnp, "b1e1", f->Derivative(a), f->Derivative(b));

      spline3->SetLineColor(kGreen);
      spline3->SetLineWidth(3);
      spline3->SetMarkerColor(kMagenta);
      spline3->SetMarkerStyle(20);
      spline3->SetMarkerSize(1.5);

      // Draw the third spline
      spline3->Draw("lcpsame");

      delete legend;
      legend = gPad->BuildLegend(0.6, 0.7, 0.88, 0.88);

      gPad->Update();

      gSystem->Sleep(500);
   }
}
