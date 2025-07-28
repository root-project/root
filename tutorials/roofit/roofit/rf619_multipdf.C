#include <RooRealVar.h>
#include <RooAbsPdf.h>
#include <RooCategory.h>
#include <RooArgList.h>
#include <RooMultiPdf.h>
#include <RooDataSet.h>
#include <RooMinimizer.h>
#include <RooAbsReal.h>
#include <RooGaussian.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <iostream>
#include <memory>
#include <vector>

using namespace RooFit;

void rf619_multipdf()
{
   RooRealVar x("x", "x", -10, 10);

   RooRealVar true_mean("true_mean", "true mean", 0);
   RooRealVar true_sigma("true_sigma", "true sigma", 1, 0.01, 5);
   RooGaussian true_pdf("true_pdf", "True Gaussian", x, true_mean, true_sigma);

   std::unique_ptr<RooDataSet> data(true_pdf.generate(x, 10));

   RooRealVar mean1("mean1", "mean1", 0, -1.5, 1.5);
   RooRealVar mean2("mean2", "mean2", 0.5, -1.5, 1.5);
   RooRealVar mean3("mean3", "mean3", 2, -1.5, 1.5);

   RooRealVar sigma1("sigma1", "sigma1", 1.11, 0.1, 5);
   RooRealVar sigma2("sigma2", "sigma2", 1.10, 0.1, 5);
   RooRealVar sigma3("sigma3", "sigma3", 1.05, 0.1, 5);

   RooGaussian gauss1("gauss1", "Gaussian 1", x, mean1, sigma1);
   RooGaussian gauss2("gauss2", "Gaussian 2", x, mean1, sigma2);
   RooGaussian gauss3("gauss3", "Gaussian 3", x, mean1, sigma3);

   mean1.setConstant(false); // at least one of the parameters should be free in order to minimizer to work

   // mean2.setConstant(true);
   // mean3.setConstant(true);
   sigma1.setConstant(true);
   sigma2.setConstant(true);
   sigma3.setConstant(true);

   RooCategory indexCat("indexCat", "Model index");

   RooArgList pdfList;
   pdfList.add(gauss1);
   pdfList.add(gauss2);
   pdfList.add(gauss3);

   RooMultiPdf multiPdf("multiPdf", "Multi-model PDF", indexCat, pdfList);

   // Fix the index to 0 (e.g. gauss1) for this example
   indexCat.setIndex(0);
   indexCat.setConstant();

   std::unique_ptr<RooAbsReal> nll(multiPdf.createNLL(*data, EvalBackend("codegen")));

   std::unique_ptr<RooAbsReal> nll2(gauss2.createNLL(*data, EvalBackend("codegen")));
   std::unique_ptr<RooAbsReal> nll3(gauss3.createNLL(*data, EvalBackend("codegen")));

   RooMinimizer minim(*nll);
   minim.setPrintLevel(1);
   minim.minimize("Minuit2", "Migrad");

   /* std::cout << "Best model index = " << indexCat.getIndex() << std::endl;
    std::cout << "Best PDF: " << multiPdf.getCurrentPdf()->GetName() << std::endl; */

   RooPlot *frame1 = mean1.frame(Title("RooMultiPdf fit to toy data"));

   nll->plotOn(frame1, LineColor(kBlack));
   nll2->plotOn(frame1, LineColor(kRed));
   nll3->plotOn(frame1, LineColor(kBlue));

   TCanvas *c1 = new TCanvas("c1", "NLL Plot", 800, 600); // NLL plot
   frame1->Draw();
   c1->Update();
   // Minimize over all pdfs
   double bestNLL = 1e30;
   int bestIndex = -1;
   RooAbsReal *bestNLLObj = nullptr;

   int nPdfs = pdfList.getSize();
   int currentIndex = 0;

   std::vector<std::unique_ptr<RooAbsReal>> nlls;

   while (currentIndex < nPdfs) {
      std::cout << "Fitting model index " << currentIndex << std::endl;

      indexCat.setIndex(currentIndex);
      indexCat.setConstant(true);

      auto currentNLL = std::unique_ptr<RooAbsReal>(multiPdf.createNLL(*data, EvalBackend("codegen")));

      RooMinimizer currentMinim(*currentNLL);
      currentMinim.setPrintLevel(0);
      currentMinim.minimize("Minuit2", "Migrad");

      double nllVal = currentNLL->getVal();
      std::cout << "NLL value for index " << currentIndex << " = " << nllVal << std::endl;

      nlls.push_back(std::move(currentNLL));

      if (nllVal < bestNLL) {
         bestNLL = nllVal;
         bestIndex = currentIndex;
         bestNLLObj = nlls.back().get();
      }

      ++currentIndex;
   }

   std::cout << "\nBrute-force Best model index = " << bestIndex << std::endl;
   std::cout << "Brute-force Best NLL value = " << bestNLL << std::endl;
}
