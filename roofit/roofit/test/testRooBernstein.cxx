// Test for RooBernstein
// Authors: Rahul Balasubramanian, CERN  10/2019

#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooBernstein.h"
#include "TCanvas.h"
#include "RooPlot.h"

#include "gtest/gtest.h"

TEST(RooBernstein, RangedFit)
{

  void runFit(unsigned int N, double a0, double a1, double a2, double a3)
  {
    RooRealVar x("x", "x", 0., 100.);
 
    // Define coefficients for a bernstein polynomial of order 3
    RooRealVar c0("c0", "c1 coeff", a0, 0., 10.);
    RooRealVar c1("c1", "c1 coeff", a1, 0., 10.);
    RooRealVar c2("c2", "c2 coeff", a2, 0., 10.);
    RooRealVar c3("c3", "c3 coeff", a3, 0., 10.);

    // Build bernstein p.d.f in terms of coefficients
    RooBernstein bern("bern", "bernstein PDF", x, RooArgList(c0, c1, c2, c3));

    // Set ranges for the variable
    x.setRange("range1", 0., 30.);
    x.setRange("range2", 70., 100.);
    x.setRange("FULL", 0., 100.);

    // Set normalization range
    bern.selectNormalizationRange("FULL",kTRUE);
   
    // Create an extended pdf to fit simultaneously in the two ranges 
    RooRealVar Ne("Ne", "number of events", 1000, -1e30, 1e30);
    RooExtendPdf extbern("extbern", "bernstein extended pdf", bern, Ne, "FULL");


    // Create a dataset from the bernstein pdf
    RooDataSet *data = bern.generate(x, N);
    data->plotOn(xframe2);
 
    // Fit pdf to data
    c0.setConstant(kTRUE);
    auto result = extbern.fitTo(*data,RooFit::Range("range1,range2"));

    // Plot the distributions
    RooPlot *xframe1 = x.frame(Title("bernstein p.d.f"));
    RooPlot *xframe2 = x.frame(Title("fitted bernstein p.d.f. with data"));
    bern.plotOn(xframe1, LineColor(kRed));
    extbern.plotOn(xframe2,LineColor(kBlue));

    // Draw all frames on a canvas
    TCanvas *c = new TCanvas("c", "c", 800, 400);
    c->Divide(2);
    c->cd(1);
    gPad->SetLeftMargin(0.15);
    xframe->GetYaxis()->SetTitleOffset(1.6);
    xframe->Draw();
    c->cd(2);
    gPad->SetLeftMargin(0.15);
    xframe2->GetYaxis()->SetTitleOffset(1.6);
    xframe2->Draw();
    c->cd(3);

    std::cout << "\nORIGINAL PARAMETER VALUES" << std::endl;
    std::cout << "c0 = " << a0 << std::endl; 
    std::cout << "c1 = " << a1 << std::endl; 
    std::cout << "c2 = " << a2 << std::endl; 
    std::cout << "c3 = " << a3 << std::endl; 
    std::cout << "N  = " << N  << std::endl;

    std::cout << "\nFIT PARAMETER VALUES" << std::endl;
    c0.Print();
    c1.Print();
    c2.Print();
    c3.Print();   
    Ne.Print();
  };
  runFit(10000, 0.3, 0.03, 0.2, 0.5);
}

