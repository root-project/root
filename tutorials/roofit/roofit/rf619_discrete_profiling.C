///\file
///\ingroup tutorial_roofit_main
/// \notebook -js
/// Basic functionality: demonstrate fitting multiple models using RooMultiPdf and selecting the best one via Discrete
/// Profiling method.
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// \date July 2025
/// \author Galin Bistrev

using namespace RooFit;

void rf619_discrete_profiling()
{
   RooRealVar x("x", "Observable", 0, 50);

   //  Category 0 Pdf-s: Exponential + Chebyshev.
   RooRealVar lambda1("lambda1", "slope1", -0.025, -0.1, -0.02);
   RooExponential expo1("expo1", "Exponential 1", x, lambda1);

   RooRealVar c0("c0", "Cheby coeff 0", -1.0, -1.0, 1.0);
   RooRealVar c1("c1", "Cheby coeff 1", 0.4, 0.05, 0.5);
   RooArgList chebCoeffs(c0, c1);
   RooChebychev cheb("cheb", "Chebyshev PDF", x, chebCoeffs);

   RooCategory pdfIndex0("pdfIndex0", "pdf index 0");
   RooMultiPdf multiPdf0("multiPdf0", "multiPdf0", pdfIndex0, RooArgList(expo1, cheb));

   // Adding complexity to the model via introdcing exponential pdf.
   RooRealVar lambdaExtra("lambdaExtra", "extra slope", -0.05, -1.0, -0.01);
   RooExponential expoExtra("expoExtra", "extra exponential", x, lambdaExtra);

   // Category 1 Pdf-s: Gaussian + Landau.
   RooRealVar mean("mean", "shared mean", 25, 0, 50);
   RooRealVar sigmaG("sigmaG", "Gaussian width", 2.0, 0.0, 5.0);
   RooRealVar sigmaL("sigmaL", "Landau width", 3.0, 1.0, 8.0);

   RooGaussian gauss1("gauss1", "Gaussian", x, mean, sigmaG);
   RooLandau landau1("landau1", "Landau", x, mean, sigmaL);

   RooCategory pdfIndex1("pdfIndex1", "pdf index 1");
   RooMultiPdf multiPdf1("multiPdf1", "multiPdf1", pdfIndex1, RooArgList(gauss1, landau1));

   // Adding complexity to the model via introdcing gaussExtra pdf. Profile scan in this model
   //   will also be done over a shared parameter from both models (mean).
   RooRealVar sigmaExtra("sigmaExtra", "extra Gaussian width", 3.0, 1.0, 6.0);
   RooGaussian gaussExtra("gaussExtra", "extra Gaussian", x, mean, sigmaExtra);

   // Creation of AddPdf objects.Add multiple PDFs into a single model for each category.
   RooRealVar frac0("frac0", "fraction for cat0", 0.7, 0.0, 1.0);
   RooAddPdf addPdf0("addPdf0", "multiPdf0 + extra expo", RooArgList(multiPdf0, gaussExtra), frac0);

   RooRealVar frac1("frac1", "fraction for cat1", 0.5, 0.0, 1.0);
   RooAddPdf addPdf1("addPdf1", "multiPdf1 + extra gauss", RooArgList(multiPdf1, expoExtra), frac1);

   // Simultaneous Pdf across categories.
   RooCategory catIndex("catIndex", "Category");
   catIndex.defineType("cat0", 0);
   catIndex.defineType("cat1", 1);

   RooSimultaneous simPdf("simPdf", "simultaneous model", catIndex);
   simPdf.addPdf(addPdf0, "cat0");
   simPdf.addPdf(addPdf1, "cat1");

   //  Generate toy data for each AddPdf.
   RooDataSet *data0 = addPdf0.generate(x, 800);
   RooDataSet *data1 = addPdf1.generate(x, 1000);

   // Ploting individual Pdf-s

   RooPlot *frame0 = x.frame();
   data0->plotOn(frame0);
   addPdf0.plotOn(frame0);
   pdfIndex0.setIndex(1);
   addPdf0.plotOn(frame0, LineColor(kRed));
   frame0->SetTitle("");
   frame0->GetXaxis()->SetTitle("Observable");
   frame0->GetYaxis()->SetTitle("Events");

   TLegend *leg0 = new TLegend(0.6, 0.7, 0.9, 0.9);
   leg0->AddEntry(frame0->getObject(0), "Data", "lep"); // Data points
   leg0->AddEntry(frame0->getObject(1), "Expo ", "l");
   leg0->AddEntry(frame0->getObject(2), "Poly", "l");

   // Ploting individual Pdf-s - Category 1

   RooPlot *frame1 = x.frame();
   data1->plotOn(frame1);
   addPdf1.plotOn(frame1);
   pdfIndex1.setIndex(1);
   addPdf1.plotOn(frame1, LineColor(kRed));
   frame1->SetTitle("");
   frame1->GetXaxis()->SetTitle("Observable");
   frame1->GetYaxis()->SetTitle("Events");

   // Create legend for Category 1
   TLegend *leg1 = new TLegend(0.6, 0.7, 0.9, 0.9);
   leg1->AddEntry(frame1->getObject(0), "Data", "lep"); // Data points
   leg1->AddEntry(frame1->getObject(1), "Gauss", "l");
   leg1->AddEntry(frame1->getObject(2), "Landau", "l");

   // Combine datasets for simultaneous fit.
   RooDataSet *data = new RooDataSet("data", "combined", RooArgSet(x, catIndex));

   RooArgSet vars(x, catIndex);

   for (int i = 0; i < data0->numEntries(); ++i) {
      x.setVal(data0->get(i)->getRealValue("x"));
      catIndex.setLabel("cat0");
      data->add(vars);
   }
   for (int i = 0; i < data1->numEntries(); ++i) {
      x.setVal(data1->get(i)->getRealValue("x"));
      catIndex.setLabel("cat1");
      data->add(vars);
   }

   // Create NLL with codegen and minimize it via the discrete profiling method.
   std::unique_ptr<RooAbsReal> nll1(simPdf.createNLL(*data, EvalBackend("codegen")));
   RooMinimizer minim(*nll1);

   minim.setStrategy(1);
   minim.setEps(1e-7);
   minim.setPrintLevel(-1);

   // Setup profiling of 'mean' over discrete combinations.
   const int nMeanPoints = 40;
   const double meanMin = 17;
   const double meanMax = 33;

   // Generate all discrete combinations of PDF indices for profiling.
   std::vector<std::vector<int>> combosToPlot;
   for (int i = 0; i < pdfIndex0.size(); ++i) {
      for (int j = 0; j < pdfIndex1.size(); ++j) {
         combosToPlot.push_back({i, j});
      }
   }

   // Creates a canvas  where all NLL vs mean  will be drawn.
   TCanvas *c = new TCanvas("c_rf619", "NLL vs Mean for Different Discrete Combinations", 1200, 400);
   c->Divide(3, 1);

   // Define arrays of ROOT colors and marker styles.
   int colors[] = {kRed, kBlue, kGreen + 2, kMagenta, kOrange + 7};
   int markers[] = {20, 21, 22, 23, 33};

   // Container to store all TGraph objects:
   std::vector<TGraph *> graphs;

   // Create one TGraph for each discrete combination of indices.
   // Each graph will hold the NLL vs mean curve for that combination.
   // Colors and marker styles are cycled through the predefined arrays.
   for (size_t idx = 0; idx < combosToPlot.size(); ++idx) {
      TGraph *g = new TGraph(nMeanPoints);
      g->SetLineColor(colors[idx % 5]);
      g->SetMarkerColor(colors[idx % 5]);
      g->SetMarkerStyle(markers[idx % 5]);
      g->SetTitle(Form("Combo [%d,%d]", combosToPlot[idx][0], combosToPlot[idx][1]));
      graphs.push_back(g);
   }

   // Create a special TGraph for the profiled NLL curve across all combinations.
   // Drawn in bold black to stand out.
   TGraph *profileGraph = new TGraph(nMeanPoints);
   profileGraph->SetLineColor(kBlack);
   profileGraph->SetLineWidth(4);
   profileGraph->SetMarkerColor(kBlack);
   profileGraph->SetMarkerStyle(22);
   profileGraph->SetTitle("Profile");
   graphs.push_back(profileGraph);

   // Loop over mean values, compute NLL and profile likelihood.
   for (int i = 0; i < nMeanPoints; ++i) {

      // Fix the "mean" parameter at this scan point
      double meanVal = meanMin + i * (meanMax - meanMin) / (nMeanPoints - 1);
      mean.setVal(meanVal);

      // Loop over all discrete PDF index combinations
      for (size_t comboIdx = 0; comboIdx < combosToPlot.size(); ++comboIdx) {
         const auto &combo = combosToPlot[comboIdx];

         // Fix both category indices to this combination
         pdfIndex0.setIndex(combo[0]);
         pdfIndex1.setIndex(combo[1]);

         // Freeze discrete indices and the mean parameter
         // so the minimizer only optimizes continuous parameters
         pdfIndex0.setConstant(true);
         pdfIndex1.setConstant(true);
         mean.setConstant(true);

         // Minimize over the continuous parameters
         // Evaluate NLL at this configuration
         minim.minimize("Minuit2", "Migrad");
         double nllVal = nll1->getVal();

         // Store NLL vs. mean value in the corresponding graph
         graphs[comboIdx]->SetPoint(i, meanVal, nllVal);

         // Unfreeze categories and mean so the next loop iteration can change thems
         pdfIndex0.setConstant(false);
         pdfIndex1.setConstant(false);
         mean.setConstant(false);
      }

      // Freeze mean (profiling at this value)
      mean.setConstant(true);

      // Scan all discrete combinations internally
      minim.minimize("Minuit2", "Migrad");

      // Store profiled NLL and plot it
      double profNLL = nll1->getVal();
      graphs.back()->SetPoint(i, meanVal, profNLL);
   }
   // Panel 1: Category 0
   c->cd(1);
   gPad->SetLeftMargin(0.15);
   frame0->GetYaxis()->SetTitleOffset(1.4);
   frame0->Draw();
   if (leg0)
      leg0->Draw();

   // Panel 2: Category 1
   c->cd(2);
   gPad->SetLeftMargin(0.15);
   frame1->GetYaxis()->SetTitleOffset(1.4);
   frame1->Draw();
   if (leg1)
      leg1->Draw();

   c->cd(3);
   gPad->SetLeftMargin(0.15);

   TMultiGraph *mg = new TMultiGraph();
   for (auto &g : graphs) {
      mg->Add(g, "PL");
      mg->GetYaxis()->SetTitleOffset(1.8);
   }

   mg->Draw("APL");
   mg->GetXaxis()->SetTitle("Mean");
   mg->GetYaxis()->SetTitle("NLL");

   gPad->BuildLegend();
}
