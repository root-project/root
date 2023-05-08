/// \file
/// \ingroup tutorial_roostats
/// \notebook
/// Produces an interval on the mean signal in a number counting experiment with known background using the
/// Feldman-Cousins technique.
///
/// Using the RooStats FeldmanCousins tool with 200 bins
/// it takes 1 min and the interval is [0.2625, 10.6125]
/// with a step size of 0.075.
/// The interval in Feldman & Cousins's original paper is [.29, 10.81] Phys.Rev.D57:3873-3889,1998.
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Kyle Cranmer

#include "RooGlobalFunc.h"
#include "RooStats/ConfInterval.h"
#include "RooStats/PointSetInterval.h"
#include "RooStats/ConfidenceBelt.h"
#include "RooStats/FeldmanCousins.h"
#include "RooStats/ModelConfig.h"

#include "RooWorkspace.h"
#include "RooDataSet.h"
#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooAddition.h"

#include "RooDataHist.h"

#include "RooPoisson.h"
#include "RooPlot.h"

#include "TCanvas.h"
#include "TTree.h"
#include "TH1F.h"
#include "TMarker.h"
#include "TStopwatch.h"

#include <iostream>

// use this order for safety on library loading
using namespace RooFit;
using namespace RooStats;

void rs401c_FeldmanCousins()
{
   // to time the macro... about 30 s
   TStopwatch t;
   t.Start();

   // make a simple model
   RooRealVar x("x", "", 1, 0, 50);
   RooRealVar mu("mu", "", 2.5, 0, 15); // with a limit on mu>=0
   RooConstVar b("b", "", 3.);
   RooAddition mean("mean", "", RooArgList(mu, b));
   RooPoisson pois("pois", "", x, mean);
   RooArgSet parameters(mu);

   // create a toy dataset
   RooDataSet *data = pois.generate(RooArgSet(x), 1);
   data->Print("v");

   TCanvas *dataCanvas = new TCanvas("dataCanvas");
   RooPlot *frame = x.frame();
   data->plotOn(frame);
   frame->Draw();
   dataCanvas->Update();

   RooWorkspace *w = new RooWorkspace();
   ModelConfig modelConfig("poissonProblem", w);
   modelConfig.SetPdf(pois);
   modelConfig.SetParametersOfInterest(parameters);
   modelConfig.SetObservables(RooArgSet(x));
   w->Print();

   //////// show use of Feldman-Cousins
   RooStats::FeldmanCousins fc(*data, modelConfig);
   fc.SetTestSize(.05); // set size of test
   fc.UseAdaptiveSampling(true);
   fc.FluctuateNumDataEntries(false); // number counting analysis: dataset always has 1 entry with N events observed
   fc.SetNBins(100);                  // number of points to test per parameter

   // use the Feldman-Cousins tool
   PointSetInterval *interval = (PointSetInterval *)fc.GetInterval();

   // make a canvas for plots
   TCanvas *intervalCanvas = new TCanvas("intervalCanvas");

   std::cout << "is this point in the interval? " << interval->IsInInterval(parameters) << std::endl;

   std::cout << "interval is [" << interval->LowerLimit(mu) << ", " << interval->UpperLimit(mu) << "]" << endl;

   // using 200 bins it takes 1 min and the answer is
   // interval is [0.2625, 10.6125] with a step size of .075
   // The interval in Feldman & Cousins's original paper is [.29, 10.81]
   //  Phys.Rev.D57:3873-3889,1998.

   // No dedicated plotting class yet, so do it by hand:

   RooDataHist *parameterScan = (RooDataHist *)fc.GetPointsToScan();
   TH1F *hist = (TH1F *)parameterScan->createHistogram("mu", Binning(30));
   hist->Draw();

   RooArgSet *tmpPoint;
   // loop over points to test
   for (Int_t i = 0; i < parameterScan->numEntries(); ++i) {
      //    cout << "on parameter point " << i << " out of " << parameterScan->numEntries() << endl;
      // get a parameter point from the list of points to test.
      tmpPoint = (RooArgSet *)parameterScan->get(i)->clone("temp");

      TMarker *mark = new TMarker(tmpPoint->getRealValue("mu"), 1, 25);
      if (interval->IsInInterval(*tmpPoint))
         mark->SetMarkerColor(kBlue);
      else
         mark->SetMarkerColor(kRed);

      mark->Draw("s");
      // delete tmpPoint;
      //    delete mark;
   }
   t.Stop();
   t.Print();
}
