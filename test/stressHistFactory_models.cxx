// ROOT headers
#include "TString.h"

// RooFit headers
#include "RooWorkspace.h"

// HistFactory headers
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/MakeModelAndMeasurementsFast.h"

using namespace RooFit;
using namespace RooStats;

//______________________________________________________________________________
void buildAPI_XML_TestModel(TString prefix)
{
  // Build model for prototype on/off problem
  // Poiss(x | s+b) * Poiss(y | tau b )
  HistFactory::Measurement meas("Test","API_XML_TestModel");

  // do not fit, just export the workspace
  meas.SetExportOnly(true);

  // put output in separate sub-directory
  meas.SetOutputFilePrefix(prefix.Data());

  // we are interested in the number of signal events
  meas.SetPOI("mu");

  // histograms are already scaled to luminosity
  // relative uncertainty of lumi is 10%, but lumi will be treated as constant later
  meas.SetLumi(1.0);
  meas.SetLumiRelErr(0.1);
  meas.AddConstantParam("Lumi");

  // create channel for signal region with observed data
  HistFactory::Channel SignalRegion("SignalRegion");
  SignalRegion.SetData("Data","HistFactory_input.root","API_vs_XML/SignalRegion/");
  SignalRegion.SetStatErrorConfig(0.05,HistFactory::Constraint::Poisson);

  // add signal sample to signal region
  HistFactory::Sample Signal("signal","signal","HistFactory_input.root","API_vs_XML/SignalRegion/");
  Signal.AddNormFactor("mu",1,0,10);
  Signal.AddOverallSys("AccSys",0.95,1.05);
  SignalRegion.AddSample(Signal);

  // add background1 sample to signal region
  HistFactory::Sample Background1("background1","background1","HistFactory_input.root","API_vs_XML/SignalRegion/");
  Background1.ActivateStatError();
  Background1.AddHistoSys("bkg1_shape_unc","background1_Low","HistFactory_input.root","API_vs_XML/SignalRegion/",
                          "background1_High","HistFactory_input.root","API_vs_XML/SignalRegion/");
  Background1.AddOverallSys("bkg_unc",0.9,1.1);
  SignalRegion.AddSample(Background1);

  // add background2 sample to signal region
  HistFactory::Sample Background2("background2","background2","HistFactory_input.root","API_vs_XML/SignalRegion/");
  Background2.SetNormalizeByTheory(kFALSE);
  Background2.AddNormFactor("bkg",1,0,20);
  Background2.AddOverallSys("bkg_unc",0.9,1.2);
  Background2.AddShapeSys("bkg2_shape_unc",HistFactory::Constraint::Gaussian,"bkg2_shape_unc","HistFactory_input.root","API_vs_XML/SignalRegion/");
  SignalRegion.AddSample(Background2);

  // create channel for sideband region with observed data
  HistFactory::Channel SidebandRegion("SidebandRegion");
  SidebandRegion.SetData("Data","HistFactory_input.root","API_vs_XML/SidebandRegion/");

  // add background sample to sideband region
  HistFactory::Sample Background3("background","unitHist","HistFactory_input.root","API_vs_XML/SidebandRegion/");
  Background3.SetNormalizeByTheory(kFALSE);
  Background3.AddNormFactor("bkg",1,0,20);
  Background3.AddNormFactor("tau",10,0.0,1000.0);
  SidebandRegion.AddSample(Background3);

  // add channels to measurement
  meas.AddChannel(SignalRegion);
  meas.AddChannel(SidebandRegion);

  // get histograms
  meas.CollectHistograms();

  // build model
  HistFactory::MakeModelAndMeasurementFast(meas);

  meas.PrintXML();
}
