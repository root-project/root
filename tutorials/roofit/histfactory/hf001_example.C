/// \file
/// \ingroup tutorial_histfactory
/// A ROOT script demonstrating  an example of writing and fitting a HistFactory model using C++ only.
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// \author George Lewis

#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>
#include <RooStats/HistFactory/Measurement.h>

#include <TFile.h>
#include <TROOT.h>

void hf001_example()
{
   std::string InputFile = "./data/example.root";
   // in case the file is not found
   bool bfile = gSystem->AccessPathName(InputFile.c_str());
   if (bfile) {
      std::cout << "Input file is not found - run prepareHistFactory script " << std::endl;
      gROOT->ProcessLine(".! prepareHistFactory .");
      bfile = gSystem->AccessPathName(InputFile.c_str());
      if (bfile) {
         std::cout << "Still no " << InputFile << ", giving up.\n";
         exit(1);
      }
   }

   // Create the measurement
   RooStats::HistFactory::Measurement meas("meas", "meas");

   meas.SetOutputFilePrefix("./results/example_UsingC");
   meas.SetPOI("SigXsecOverSM");
   meas.AddConstantParam("alpha_syst1");
   meas.AddConstantParam("Lumi");

   meas.SetLumi(1.0);
   meas.SetLumiRelErr(0.10);
   meas.SetBinHigh(2);

   // Create a channel

   RooStats::HistFactory::Channel chan("channel1");
   chan.SetData("data", InputFile);
   chan.SetStatErrorConfig(0.05, "Poisson");

   // Now, create some samples

   // Create the signal sample
   RooStats::HistFactory::Sample signal("signal", "signal", InputFile);
   signal.AddOverallSys("syst1", 0.95, 1.05);
   signal.AddNormFactor("SigXsecOverSM", 1, 0, 3);
   chan.AddSample(signal);

   // Background 1
   RooStats::HistFactory::Sample background1("background1", "background1", InputFile);
   background1.ActivateStatError("background1_statUncert", InputFile);
   background1.AddOverallSys("syst2", 0.95, 1.05);
   chan.AddSample(background1);

   // Background 1
   RooStats::HistFactory::Sample background2("background2", "background2", InputFile);
   background2.ActivateStatError();
   background2.AddOverallSys("syst3", 0.95, 1.05);
   chan.AddSample(background2);

   // Done with this channel
   // Add it to the measurement:
   meas.AddChannel(chan);

   // Collect the histograms from their files,
   // print some output,
   meas.CollectHistograms();
   meas.PrintTree();

   // One can print XML code to an output directory:
   // meas.PrintXML( "xmlFromCCode", meas.GetOutputFilePrefix() );

   // Now, do the measurement
   std::unique_ptr<RooWorkspace> ws{MakeModelAndMeasurementFast(meas)};

   RooStats::ModelConfig *modelConfig = static_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));

   // Get probability density function and parameters list from model
   RooAbsPdf *pdf = modelConfig->GetPdf();
   RooArgSet globalObservables{*modelConfig->GetGlobalObservables()};

   // Perform the fit using Minos to get the correct asymmetric uncertainties
   using namespace RooFit;
   std::unique_ptr<RooFitResult> result{
      pdf->fitTo(*ws->data("obsData"), Save(), PrintLevel(-1), GlobalObservables(globalObservables), Minos(true))};

   // Getting list of Parameters of Interest and getting first from them
   RooRealVar *poi = static_cast<RooRealVar *>(modelConfig->GetParametersOfInterest()->first());

   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*ws->data("obsData"))};
   std::unique_ptr<RooAbsReal> profile{nll->createProfile(*poi)};

   // frame for future plot
   RooPlot *frame = poi->frame();

   frame->SetTitle("");
   frame->GetYaxis()->SetTitle("-log likelihood");
   frame->GetXaxis()->SetTitle(poi->GetTitle());

   TCanvas *profileLikelihoodCanvas = new TCanvas{"combined", "", 800, 600};

   double xmin = poi->getMin();
   double xmax = poi->getMax();
   TLine *line = new TLine(xmin, .5, xmax, .5);
   line->SetLineColor(kGreen);
   TLine *line90 = new TLine(xmin, 2.71 / 2., xmax, 2.71 / 2.);
   line90->SetLineColor(kGreen);
   TLine *line95 = new TLine(xmin, 3.84 / 2., xmax, 3.84 / 2.);
   line95->SetLineColor(kGreen);
   frame->addObject(line);
   frame->addObject(line90);
   frame->addObject(line95);

   nll->plotOn(frame, ShiftToZero(), LineColor(kRed), LineStyle(kDashed));
   profile->plotOn(frame);

   frame->SetMinimum(0);
   frame->SetMaximum(2.);

   frame->Draw();

   // Print fit results to console in verbose mode to see asymmetric uncertainties
   result->Print("v");
}
