///\file
///\ingroup tutorial_roofit_main
/// \notebook -js
/// Basic functionality: demonstrate fitting multiple models using RooMultiPdf and selecting the best one via NLL comparison.
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// \date July 2025
/// \author Galin Bistrev



#include <RooAbsPdf.h>      
#include <RooAbsReal.h>    
#include <RooArgList.h>    
#include <RooCategory.h>    
#include <RooDataSet.h>     
#include <RooGaussian.h>    
#include <RooMinimizer.h>  
#include <RooMultiPdf.h>    // Container for multiple PDFs
#include <RooRealVar.h>     
#include <TCanvas.h>       
#include <TLegend.h>       
#include <iostream>         
#include <memory>           
#include <vector> 


void rf619_multipdf()
{
// Define observable variable x
   RooRealVar x("x", "x", -10, 10);

   // Define "true" Gaussian model parameters and PDF
   RooRealVar true_mean("true_mean", "true mean", 0);
   RooRealVar true_sigma("true_sigma", "true sigma", 1, 0.01, 5);
   RooGaussian true_pdf("true_pdf", "True Gaussian", x, true_mean, true_sigma);

   // Generate toy data from the true PDF
   std::unique_ptr<RooDataSet> data(true_pdf.generate(x, 10));

   // Define means and sigmas for candidate models;mean2 and mean3 can be used as a backup 
   // if minimization with mean1 doesnt yield meaningful result
   RooRealVar mean1("mean1", "mean1", 0, -1.5, 1.5);
   RooRealVar mean2("mean2", "mean2", 0.5, -1.5, 1.5);   
   RooRealVar mean3("mean3", "mean3", 2, -1.5, 1.5);     

   RooRealVar sigma1("sigma1", "sigma1", 1.11, 0.1, 5);
   RooRealVar sigma2("sigma2", "sigma2", 1.10, 0.1, 5);
   RooRealVar sigma3("sigma3", "sigma3", 1.05, 0.1, 5);

   // Define three Gaussian models (all sharing the same mean)
   RooGaussian gauss1("gauss1", "Gaussian 1", x, mean1, sigma1);
   RooGaussian gauss2("gauss2", "Gaussian 2", x, mean1, sigma2);
   RooGaussian gauss3("gauss3", "Gaussian 3", x, mean1, sigma3);

   mean1.setConstant(false); // Ensure at least one free parameter for minimizer to work

   // Fix sigmas to distinguish models
   sigma1.setConstant(true);
   sigma2.setConstant(true);
   sigma3.setConstant(true);

   // Create a category to index the models in RooMultiPdf and add them to the argument list
   RooCategory indexCat("indexCat", "Model index");


   RooArgList pdfList;
   pdfList.add(gauss1);
   pdfList.add(gauss2);
   pdfList.add(gauss3);

   // Construct the RooMultiPdf with the list of models
   RooMultiPdf multiPdf("multiPdf", "Multi-model PDF", indexCat, pdfList);

   // Fix model index to 0 (i.e., gauss1) and create NLL for plotting for gauss2 and gauss3
   indexCat.setIndex(0);
   indexCat.setConstant();

   std::unique_ptr<RooAbsReal> nll(multiPdf.createNLL(*data, EvalBackend("codegen")));
   std::unique_ptr<RooAbsReal> nll2(gauss2.createNLL(*data, EvalBackend("codegen")));
   std::unique_ptr<RooAbsReal> nll3(gauss3.createNLL(*data, EvalBackend("codegen")));

   // Fit the fixed model (index 0) to the data
   RooMinimizer minim(*nll);
   minim.setPrintLevel(1);
   minim.minimize("Minuit2", "Migrad");

   // Plot the NLLs of each model on a shared frame
   RooPlot *frame1 = mean1.frame(Title("RooMultiPdf fit to toy data"));
   nll->plotOn(frame1, LineColor(kBlack));
   nll2->plotOn(frame1, LineColor(kRed));
   nll3->plotOn(frame1, LineColor(kBlue));

   // Display the NLL plot
   TCanvas *c1 = new TCanvas("c1", "NLL Plot", 800, 600);
   frame1->Draw();
   c1->Update();

   // Brute-force loop: find the best-fitting model by looping over all
   double bestNLL = 1e30;
   int bestIndex = -1;
   RooAbsReal *bestNLLObj = nullptr;

   int nPdfs = pdfList.getSize();
   int currentIndex = 0;
   std::vector<std::unique_ptr<RooAbsReal>> nlls;

   while (currentIndex < nPdfs) {
      std::cout << "Fitting model index " << currentIndex << std::endl;

      indexCat.setIndex(currentIndex); // Select model
      indexCat.setConstant(true);

      // Create and minimize the NLL for current model
      auto currentNLL = std::unique_ptr<RooAbsReal>(multiPdf.createNLL(*data, EvalBackend("codegen")));
      RooMinimizer currentMinim(*currentNLL);
      currentMinim.setPrintLevel(0);
      currentMinim.minimize("Minuit2", "Migrad");

      double nllVal = currentNLL->getVal();
      std::cout << "NLL value for index " << currentIndex << " = " << nllVal << std::endl;

      nlls.push_back(std::move(currentNLL));

      // Keep track of best model
      if (nllVal < bestNLL) {
         bestNLL = nllVal;
         bestIndex = currentIndex;
         bestNLLObj = nlls.back().get();
      }

      ++currentIndex;
   }

   // Print the best-fit model index and its NLL value
   std::cout << "\nBrute-force Best model index = " << bestIndex << std::endl;
   std::cout << "Brute-force Best NLL value = " << bestNLL << std::endl;
}