/// \file
/// \ingroup tutorial_roostats
/// \notebook
/// Standard demo of the Profile Likelihood calculator
/// StandardProfileLikelihoodDemo
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
/// The ProfileLikelihoodCalculator is based on Wilks's theorem
/// and the asymptotic properties of the profile likelihood ratio
/// (eg. that it is chi-square distributed for the true value).
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Kyle Cranmer

#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "RooWorkspace.h"
#include "RooAbsData.h"
#include "RooRealVar.h"

#include "RooStats/ModelConfig.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodInterval.h"
#include "RooStats/LikelihoodIntervalPlot.h"

using namespace RooFit;
using namespace RooStats;

struct ProfileLikelihoodOptions {

   double confLevel = 0.95;
   int nScanPoints = 50;
   bool plotAsTF1 = false;
   double poiMinPlot = 1;
   double poiMaxPlot = 0;
   bool doHypoTest = false;
   double nullValue = 0;
};

ProfileLikelihoodOptions optPL;

void StandardProfileLikelihoodDemo(const char *infile = "", const char *workspaceName = "combined",
                                   const char *modelConfigName = "ModelConfig", const char *dataName = "obsData")
{

   double confLevel = optPL.confLevel;
   double nScanPoints = optPL.nScanPoints;
   bool plotAsTF1 = optPL.plotAsTF1;
   double poiXMin = optPL.poiMinPlot;
   double poiXMax = optPL.poiMaxPlot;
   bool doHypoTest = optPL.doHypoTest;
   double nullParamValue = optPL.nullValue;

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

   // ---------------------------------------------
   // create and use the ProfileLikelihoodCalculator
   // to find and plot the 95% confidence interval
   // on the parameter of interest as specified
   // in the model config
   ProfileLikelihoodCalculator pl(*data, *mc);
   pl.SetConfidenceLevel(confLevel); // 95% interval
   LikelihoodInterval *interval = pl.GetInterval();

   // print out the interval on the first Parameter of Interest
   RooRealVar *firstPOI = (RooRealVar *)mc->GetParametersOfInterest()->first();
   cout << "\n>>>> RESULT : " << confLevel * 100 << "% interval on " << firstPOI->GetName() << " is : ["
        << interval->LowerLimit(*firstPOI) << ", " << interval->UpperLimit(*firstPOI) << "]\n " << endl;

   // make a plot

   cout << "making a plot of the profile likelihood function ....(if it is taking a lot of time use less points or the "
           "TF1 drawing option)\n";
   LikelihoodIntervalPlot plot(interval);
   plot.SetNPoints(nScanPoints); // do not use too many points, it could become very slow for some models
   if (poiXMin < poiXMax)
      plot.SetRange(poiXMin, poiXMax);
   TString opt;
   if (plotAsTF1)
      opt += TString("tf1");
   plot.Draw(opt); // use option TF1 if too slow (plot.Draw("tf1")

   // if requested perform also an hypothesis test for the significance
   if (doHypoTest) {
      RooArgSet nullparams("nullparams");
      nullparams.addClone(*firstPOI);
      nullparams.setRealValue(firstPOI->GetName(), nullParamValue);
      pl.SetNullParameters(nullparams);
      std::cout << "Perform Test of Hypothesis : null Hypothesis is " << firstPOI->GetName() << nullParamValue
                << std::endl;
      auto result = pl.GetHypoTest();
      std::cout << "\n>>>> Hypotheis Test Result \n";
      result->Print();
   }
}
