/// \file
/// \ingroup tutorial_roostats
/// \notebook
/// \brief Comparison of MCMC and PLC in a multi-variate gaussian problem
///
/// This tutorial produces an N-dimensional multivariate Gaussian
/// with a non-trivial covariance matrix.  By default N=4 (called "dim").
///
/// A subset of these are considered parameters of interest.
/// This problem is tractable analytically.
///
/// We use this mainly as a test of Markov Chain Monte Carlo
/// and we compare the result to the profile likelihood ratio.
///
/// We use the proposal helper to create a customized
/// proposal function for this problem.
///
/// For N=4 and 2 parameters of interest it takes about 10-20 seconds
/// and the acceptance rate is 37%
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \authors Kevin Belasco and Kyle Cranmer

#include "RooGlobalFunc.h"
#include <stdlib.h>
#include "TMatrixDSym.h"
#include "RooMultiVarGaussian.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "RooAbsReal.h"
#include "RooFitResult.h"
#include "TStopwatch.h"
#include "RooStats/MCMCCalculator.h"
#include "RooStats/MetropolisHastings.h"
#include "RooStats/MarkovChain.h"
#include "RooStats/ConfInterval.h"
#include "RooStats/MCMCInterval.h"
#include "RooStats/MCMCIntervalPlot.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/ProposalHelper.h"
#include "RooStats/ProposalFunction.h"
#include "RooStats/PdfProposal.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooStats/LikelihoodInterval.h"

using namespace std;
using namespace RooFit;
using namespace RooStats;

void MultivariateGaussianTest(Int_t dim = 4, Int_t nPOI = 2)
{
   // let's time this challenging example
   TStopwatch t;
   t.Start();

   RooArgList xVec;
   RooArgList muVec;
   RooArgSet poi;

   // make the observable and means
   Int_t i, j;
   RooRealVar *x;
   RooRealVar *mu_x;
   for (i = 0; i < dim; i++) {
      char *name = Form("x%d", i);
      x = new RooRealVar(name, name, 0, -3, 3);
      xVec.add(*x);

      char *mu_name = Form("mu_x%d", i);
      mu_x = new RooRealVar(mu_name, mu_name, 0, -2, 2);
      muVec.add(*mu_x);
   }

   // put them into the list of parameters of interest
   for (i = 0; i < nPOI; i++) {
      poi.add(*muVec.at(i));
   }

   // make a covariance matrix that is all 1's
   TMatrixDSym cov(dim);
   for (i = 0; i < dim; i++) {
      for (j = 0; j < dim; j++) {
         if (i == j)
            cov(i, j) = 3.;
         else
            cov(i, j) = 1.0;
      }
   }

   // now make the multivariate Gaussian
   RooMultiVarGaussian mvg("mvg", "mvg", xVec, muVec, cov);

   // --------------------
   // make a toy dataset
   RooDataSet *data = mvg.generate(xVec, 100);

   // now create the model config for this problem
   RooWorkspace *w = new RooWorkspace("MVG");
   ModelConfig modelConfig(w);
   modelConfig.SetPdf(mvg);
   modelConfig.SetParametersOfInterest(poi);

   // -------------------------------------------------------
   // Setup calculators

   // MCMC
   // we want to setup an efficient proposal function
   // using the covariance matrix from a fit to the data
   RooFitResult *fit = mvg.fitTo(*data, Save(true));
   ProposalHelper ph;
   ph.SetVariables((RooArgSet &)fit->floatParsFinal());
   ph.SetCovMatrix(fit->covarianceMatrix());
   ph.SetUpdateProposalParameters(true);
   ph.SetCacheSize(100);
   ProposalFunction *pdfProp = ph.GetProposalFunction();

   // now create the calculator
   MCMCCalculator mc(*data, modelConfig);
   mc.SetConfidenceLevel(0.95);
   mc.SetNumBurnInSteps(100);
   mc.SetNumIters(10000);
   mc.SetNumBins(50);
   mc.SetProposalFunction(*pdfProp);

   MCMCInterval *mcInt = mc.GetInterval();
   RooArgList *poiList = mcInt->GetAxes();

   // now setup the profile likelihood calculator
   ProfileLikelihoodCalculator plc(*data, modelConfig);
   plc.SetConfidenceLevel(0.95);
   LikelihoodInterval *plInt = (LikelihoodInterval *)plc.GetInterval();

   // make some plots
   MCMCIntervalPlot mcPlot(*mcInt);

   TCanvas *c1 = new TCanvas();
   mcPlot.SetLineColor(kGreen);
   mcPlot.SetLineWidth(2);
   mcPlot.Draw();

   LikelihoodIntervalPlot plPlot(plInt);
   plPlot.Draw("same");

   if (poiList->getSize() == 1) {
      RooRealVar *p = (RooRealVar *)poiList->at(0);
      Double_t ll = mcInt->LowerLimit(*p);
      Double_t ul = mcInt->UpperLimit(*p);
      cout << "MCMC interval: [" << ll << ", " << ul << "]" << endl;
   }

   if (poiList->getSize() == 2) {
      RooRealVar *p0 = (RooRealVar *)poiList->at(0);
      RooRealVar *p1 = (RooRealVar *)poiList->at(1);
      TCanvas *scatter = new TCanvas();
      Double_t ll = mcInt->LowerLimit(*p0);
      Double_t ul = mcInt->UpperLimit(*p0);
      cout << "MCMC interval on p0: [" << ll << ", " << ul << "]" << endl;
      ll = mcInt->LowerLimit(*p0);
      ul = mcInt->UpperLimit(*p0);
      cout << "MCMC interval on p1: [" << ll << ", " << ul << "]" << endl;

      // MCMC interval on p0: [-0.2, 0.6]
      // MCMC interval on p1: [-0.2, 0.6]

      mcPlot.DrawChainScatter(*p0, *p1);
      scatter->Update();
   }

   t.Print();
}
