/// \file
/// \ingroup tutorial_roostats
/// \notebook -js
/// \brief Standard demo of the numerical Bayesian calculator
///
/// This is a standard demo that can be used with any ROOT file
/// prepared in the standard way.  You specify:
///  - name for input ROOT file
///  - name of workspace inside ROOT file that holds model and data
///  - name of ModelConfig that specifies details for calculator tools
///  - name of dataset
///
/// With default parameters the macro will attempt to run the
/// standard hist2workspace example and read the ROOT file
/// that it produces.
///
/// The actual heart of the demo is only about 10 lines long.
///
/// The BayesianCalculator is based on Bayes's theorem
/// and performs the integration using ROOT's numeric integration utilities
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Kyle Cranmer

#include "TFile.h"
#include "TROOT.h"
#include "RooWorkspace.h"
#include "RooAbsData.h"
#include "RooRealVar.h"

#include "RooUniform.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/BayesianCalculator.h"
#include "RooStats/SimpleInterval.h"
#include "RooStats/RooStatsUtils.h"
#include "RooPlot.h"
#include "TSystem.h"

#include <cassert>

using namespace RooFit;
using namespace RooStats;

struct BayesianNumericalOptions {

   double confLevel = 0.95;      // interval CL
   TString integrationType = ""; // integration Type (default is adaptive (numerical integration)
   // possible values are "TOYMC" (toy MC integration, work when nuisances have a constraints pdf)
   //  "VEGAS" , "MISER", or "PLAIN"  (these are all possible MC integration)
   int nToys =
      10000; // number of toys used for the MC integrations - for Vegas should be probably set to an higher value
   bool scanPosterior =
      false; // flag to compute interval by scanning posterior (it is more robust but maybe less precise)
   bool plotPosterior = false; // plot posterior function after having computed the interval
   int nScanPoints = 50; // number of points for scanning the posterior (if scanPosterior = false it is used only for
                         // plotting). Use by default a low value to speed-up tutorial
   int intervalType = 1; // type of interval (0 is shortest, 1 central, 2 upper limit)
   double maxPOI = -999; // force a different value of POI for doing the scan (default is given value)
   double nSigmaNuisance = -1; // force integration of nuisance parameters to be within nSigma of their error (do first
                               // a model fit to find nuisance error)
};

BayesianNumericalOptions optBayes;

void StandardBayesianNumericalDemo(const char *infile = "", const char *workspaceName = "combined",
                                   const char *modelConfigName = "ModelConfig", const char *dataName = "obsData")
{

   // option definitions
   double confLevel = optBayes.confLevel;
   TString integrationType = optBayes.integrationType;
   int nToys = optBayes.nToys;
   bool scanPosterior = optBayes.scanPosterior;
   bool plotPosterior = optBayes.plotPosterior;
   int nScanPoints = optBayes.nScanPoints;
   int intervalType = optBayes.intervalType;
   int maxPOI = optBayes.maxPOI;
   double nSigmaNuisance = optBayes.nSigmaNuisance;

   // -------------------------------------------------------
   // First part is just to access a user-defined file
   // or create the standard example file if it doesn't exist

   const char *filename = "";
   if (!strcmp(infile, "")) {
      filename = "results/example_combined_GaussExample_model.root";
      bool fileExist = !gSystem->AccessPathName(filename); // note opposite return code
      // if file does not exists generate with histfactory
      if (!fileExist) {
#ifdef _WIN32
         cout << "HistFactory file cannot be generated on Windows - exit" << endl;
         return;
#endif
         // Normally this would be run on the command line
         cout << "will run standard hist2workspace example" << endl;
         gROOT->ProcessLine(".! prepareHistFactory .");
         gROOT->ProcessLine(".! hist2workspace config/example.xml");
         cout << "\n\n---------------------" << endl;
         cout << "Done creating example input" << endl;
         cout << "---------------------\n\n" << endl;
      }

   } else
      filename = infile;

   // Try to open the file
   TFile *file = TFile::Open(filename);

   // if input file was specified byt not found, quit
   if (!file) {
      cout << "StandardRooStatsDemoMacro: Input file " << filename << " is not found" << endl;
      return;
   }

   // -------------------------------------------------------
   // Tutorial starts here
   // -------------------------------------------------------

   // get the workspace out of the file
   RooWorkspace *w = (RooWorkspace *)file->Get(workspaceName);
   if (!w) {
      cout << "workspace not found" << endl;
      return;
   }

   // get the modelConfig out of the file
   ModelConfig *mc = (ModelConfig *)w->obj(modelConfigName);

   // get the modelConfig out of the file
   RooAbsData *data = w->data(dataName);

   // make sure ingredients are found
   if (!data || !mc) {
      w->Print();
      cout << "data or ModelConfig was not found" << endl;
      return;
   }

   // ------------------------------------------
   // create and use the BayesianCalculator
   // to find and plot the 95% credible interval
   // on the parameter of interest as specified
   // in the model config

   // before we do that, we must specify our prior
   // it belongs in the model config, but it may not have
   // been specified
   RooUniform prior("prior", "", *mc->GetParametersOfInterest());
   w->import(prior);
   mc->SetPriorPdf(*w->pdf("prior"));

   // do without systematics
   // mc->SetNuisanceParameters(RooArgSet() );
   if (nSigmaNuisance > 0) {
      RooAbsPdf *pdf = mc->GetPdf();
      assert(pdf);
      RooFitResult *res =
         pdf->fitTo(*data, Save(true), Minimizer(ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str()),
                    Hesse(true), PrintLevel(ROOT::Math::MinimizerOptions::DefaultPrintLevel() - 1));

      res->Print();
      RooArgList nuisPar(*mc->GetNuisanceParameters());
      for (int i = 0; i < nuisPar.getSize(); ++i) {
         RooRealVar *v = dynamic_cast<RooRealVar *>(&nuisPar[i]);
         assert(v);
         v->setMin(TMath::Max(v->getMin(), v->getVal() - nSigmaNuisance * v->getError()));
         v->setMax(TMath::Min(v->getMax(), v->getVal() + nSigmaNuisance * v->getError()));
         std::cout << "setting interval for nuisance  " << v->GetName() << " : [ " << v->getMin() << " , "
                   << v->getMax() << " ]" << std::endl;
      }
   }

   BayesianCalculator bayesianCalc(*data, *mc);
   bayesianCalc.SetConfidenceLevel(confLevel); // 95% interval

   // default of the calculator is central interval.  here use shortest , central or upper limit depending on input
   // doing a shortest interval might require a longer time since it requires a scan of the posterior function
   if (intervalType == 0)
      bayesianCalc.SetShortestInterval(); // for shortest interval
   if (intervalType == 1)
      bayesianCalc.SetLeftSideTailFraction(0.5); // for central interval
   if (intervalType == 2)
      bayesianCalc.SetLeftSideTailFraction(0.); // for upper limit

   if (!integrationType.IsNull()) {
      bayesianCalc.SetIntegrationType(integrationType); // set integrationType
      bayesianCalc.SetNumIters(nToys); // set number of iterations (i.e. number of toys for MC integrations)
   }

   // in case of toyMC make a nuisance pdf
   if (integrationType.Contains("TOYMC")) {
      RooAbsPdf *nuisPdf = RooStats::MakeNuisancePdf(*mc, "nuisance_pdf");
      cout << "using TOYMC integration: make nuisance pdf from the model " << std::endl;
      nuisPdf->Print();
      bayesianCalc.ForceNuisancePdf(*nuisPdf);
      scanPosterior = true; // for ToyMC the posterior is scanned anyway so used given points
   }

   // compute interval by scanning the posterior function
   if (scanPosterior)
      bayesianCalc.SetScanOfPosterior(nScanPoints);

   RooRealVar *poi = (RooRealVar *)mc->GetParametersOfInterest()->first();
   if (maxPOI != -999 && maxPOI > poi->getMin())
      poi->setMax(maxPOI);

   SimpleInterval *interval = bayesianCalc.GetInterval();

   // print out the interval on the first Parameter of Interest
   cout << "\n>>>> RESULT : " << confLevel * 100 << "% interval on " << poi->GetName() << " is : ["
        << interval->LowerLimit() << ", " << interval->UpperLimit() << "] " << endl;

   // end in case plotting is not requested
   if (!plotPosterior)
      return;

   // make a plot
   // since plotting may take a long time (it requires evaluating
   // the posterior in many points) this command will speed up
   // by reducing the number of points to plot - do 50

   // ignore errors of PDF if is zero
   RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::Ignore);

   cout << "\nDrawing plot of posterior function....." << endl;

   // always plot using numer of scan points
   bayesianCalc.SetScanOfPosterior(nScanPoints);

   RooPlot *plot = bayesianCalc.GetPosteriorPlot();
   plot->Draw();
}
