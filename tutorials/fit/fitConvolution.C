/// \file
/// \ingroup tutorial_fit
/// \notebook -js
/// Tutorial for convolution of two functions
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Aurelie Flandi

#include <TCanvas.h>
#include <TRandom.h>
#include <TF1Convolution.h>
#include <TF1.h>
#include <TH1F.h>

void fitConvolution()
{
   // Construction of histogram to fit.
   TH1F *h_ExpGauss = new TH1F("h_ExpGauss", "Exponential convoluted by Gaussian", 100, 0., 5.);
   for (int i = 0; i < 1e6; i++) {
      // Gives a alpha of -0.3 in the exp.
      double x = gRandom->Exp(1. / 0.3);
      x += gRandom->Gaus(0., 3.);
      // Probability density function of the addition of two variables is the
      // convolution of two density functions.
      h_ExpGauss->Fill(x);
   }

   TF1Convolution *f_conv = new TF1Convolution("expo", "gaus", -1, 6, true);
   f_conv->SetRange(-1., 6.);
   f_conv->SetNofPointsFFT(1000);
   TF1 *f = new TF1("f", *f_conv, 0., 5., f_conv->GetNpar());
   f->SetParameters(1., -0.3, 0., 1.);

   // Fit.
   new TCanvas("c", "c", 800, 1000);
   h_ExpGauss->Fit("f");
   h_ExpGauss->Draw();
}
