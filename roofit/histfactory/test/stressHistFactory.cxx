/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>
#include <RooStats/HistFactory/Measurement.h>

#include <RooStats/RooStatsUtils.h>

#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooDataSet.h>
#include <RooFit/ModelConfig.h>
#include <RooHelpers.h>
#include <RooLinkedListIter.h>
#include <RooWorkspace.h>

#include <TFile.h>
#include <TH1F.h>
#include <TMath.h>
#include <TMinuit.h>
#include <TString.h>
#include <TSystem.h>

#include <cassert>
#include <iostream>
#include <cstdio>

#include <gtest/gtest.h>

////////////////////////////////////////////////////////////////////////////////
/// Build model for prototype on/off problem
/// Poiss(x | s+b) * Poiss(y | tau b )

void buildAPI_XML_TestModel(TString prefix)
{
   using namespace RooFit;
   using namespace RooStats;

   HistFactory::Measurement meas("Test", "API_XML_TestModel");

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
   SignalRegion.SetData("Data", "HistFactory_input.root", "API_vs_XML/SignalRegion/");
   SignalRegion.SetStatErrorConfig(0.05, HistFactory::Constraint::Poisson);

   // add signal sample to signal region
   HistFactory::Sample Signal("signal", "signal", "HistFactory_input.root", "API_vs_XML/SignalRegion/");
   Signal.AddNormFactor("mu", 1, 0, 10);
   Signal.AddOverallSys("AccSys", 0.95, 1.05);
   SignalRegion.AddSample(Signal);

   // add background1 sample to signal region
   HistFactory::Sample Background1("background1", "background1", "HistFactory_input.root", "API_vs_XML/SignalRegion/");
   Background1.ActivateStatError();
   Background1.AddHistoSys("bkg1_shape_unc", "background1_Low", "HistFactory_input.root", "API_vs_XML/SignalRegion/",
                           "background1_High", "HistFactory_input.root", "API_vs_XML/SignalRegion/");
   Background1.AddOverallSys("bkg_unc", 0.9, 1.1);
   SignalRegion.AddSample(Background1);

   // add background2 sample to signal region
   HistFactory::Sample Background2("background2", "background2", "HistFactory_input.root", "API_vs_XML/SignalRegion/");
   Background2.SetNormalizeByTheory(false);
   Background2.AddNormFactor("bkg", 1, 0, 20);
   Background2.AddOverallSys("bkg_unc", 0.9, 1.2);
   Background2.AddShapeSys("bkg2_shape_unc", HistFactory::Constraint::Gaussian, "bkg2_shape_unc",
                           "HistFactory_input.root", "API_vs_XML/SignalRegion/");
   SignalRegion.AddSample(Background2);

   // create channel for sideband region with observed data
   HistFactory::Channel SidebandRegion("SidebandRegion");
   SidebandRegion.SetData("Data", "HistFactory_input.root", "API_vs_XML/SidebandRegion/");

   // add background sample to sideband region
   HistFactory::Sample Background3("background", "unitHist", "HistFactory_input.root", "API_vs_XML/SidebandRegion/");
   Background3.SetNormalizeByTheory(false);
   Background3.AddNormFactor("bkg", 1, 0, 20);
   Background3.AddNormFactor("tau", 10, 0.0, 1000.0);
   SidebandRegion.AddSample(Background3);

   // add channels to measurement
   meas.AddChannel(SignalRegion);
   meas.AddChannel(SidebandRegion);

   // get histograms
   meas.CollectHistograms();

   // build model
   HistFactory::MakeModelAndMeasurementFast(meas);

   // meas.PrintXML();
}

// PDF comparison for HistFactory
class PdfComparison {
private:
   TString fOldDirectory; // old directory where test is started
   Double_t fTolerance = 1e-3;
   int _verb = 0;

public:
   PdfComparison(Int_t verbose = 0) : fOldDirectory(gSystem->pwd()), _verb{verbose}
   {

      bool ret = gSystem->Exec("tar -xf HistFactoryTest.tar") == 0;
      if (!ret)
         Error("PdfComparison", "Error unpacking test file HistFactoryTest.tar");
   }

   bool testCode()
   {
      using namespace RooFit;
      using namespace RooStats;

      // build model using the API
      // create and move to a temporary directory to run hist2workspace
      gSystem->ChangeDirectory("API/");
      buildAPI_XML_TestModel("API_XML_TestModel");

      // get API workspace and ModelConfig
      TFile *pAPIFile = TFile::Open("API_XML_TestModel_combined_Test_model.root");
      if (!pAPIFile || pAPIFile->IsZombie()) {
         Error("testCode", "Error opening the file API_XML_TestModel_combined_Test_model.root");
         return false;
      }

      RooWorkspace *pWS_API = (RooWorkspace *)pAPIFile->Get("combined");
      if (!pWS_API) {
         Error("testCode", "Error retrieving the workspace combined");
         return false;
      }

      ModelConfig *pMC_API = (ModelConfig *)pWS_API->obj("ModelConfig");
      if (!pMC_API) {
         Error("testCode", "Error retrieving the ModelConfig");
         return false;
      }
      // build model using XML files
      gSystem->ChangeDirectory("../XML/");
      // be sure libraries are found for running hist2workspace
      gSystem->AddDynamicPath("$ROOTSYS/lib");
      TString cmd = "$ROOTSYS/bin/hist2workspace config/Measurement.xml";
      int ret = gSystem->Exec(cmd);
      if (ret != 0) {
         Error("testCode", "Error running hist2workspace");
         return false;
      }

      // get XML workspace and ModelConfig
      TFile *pXMLFile = TFile::Open("results/API_XML_TestModel_combined_Test_model.root");
      if (!pXMLFile || pXMLFile->IsZombie()) {
         Error("testCode", "Error opening the file results/API_XML_TestModel_combined_Test_model.root");
         return false;
      }

      RooWorkspace *pWS_XML = (RooWorkspace *)pXMLFile->Get("combined");
      if (!pWS_XML) {
         Error("testCode", "Error retrieving the workspace combined");
         return false;
      }

      ModelConfig *pMC_XML = (ModelConfig *)pWS_XML->obj("ModelConfig");
      if (!pMC_XML) {
         Error("testCode", "Error retrieving the ModelConfig");
         return false;
      }

      // change working directory to original one
      gSystem->ChangeDirectory(fOldDirectory);

      // compare data
      if (pWS_API->data("obsData")) {
         assert(pWS_XML->data("obsData"));
         if (!CompareData(*pWS_API->data("obsData"), *pWS_XML->data("obsData")))
            return false;
      } else
         return false;

      if (pWS_API->data("asimovData")) {
         assert(pWS_XML->data("asimovData"));
         if (!CompareData(*pWS_API->data("asimovData"), *pWS_XML->data("asimovData")))
            return false;
      } else
         return false;

      // compare sets of parameters
      if (pMC_API->GetParametersOfInterest()) {
         assert(pMC_XML->GetParametersOfInterest());
         if (_verb > 0)
            Info("testCode", "comparing PoIs");
         if (!CompareParameters(*pMC_API->GetParametersOfInterest(), *pMC_XML->GetParametersOfInterest()))
            return false;
      } else
         assert(!pMC_XML->GetParametersOfInterest());

      if (pMC_API->GetObservables()) {
         assert(pMC_XML->GetObservables());
         if (_verb > 0)
            Info("testCode", "comparing observables");
         if (!CompareParameters(*pMC_API->GetObservables(), *pMC_XML->GetObservables()))
            return false;
      } else
         assert(!pMC_XML->GetObservables());

      if (pMC_API->GetGlobalObservables()) {
         assert(pMC_XML->GetGlobalObservables());
         if (_verb > 0)
            Info("testCode", "comparing global observables");
         if (!CompareParameters(*pMC_API->GetGlobalObservables(), *pMC_XML->GetGlobalObservables()))
            return false;
      } else
         assert(!pMC_XML->GetGlobalObservables());

      if (pMC_API->GetConditionalObservables()) {
         assert(pMC_XML->GetConditionalObservables());
         if (_verb > 0)
            Info("testCode", "comparing conditional observables");
         if (!CompareParameters(*pMC_API->GetConditionalObservables(), *pMC_XML->GetConditionalObservables()))
            return false;
      } else
         assert(!pMC_XML->GetConditionalObservables());

      if (pMC_API->GetNuisanceParameters()) {
         assert(pMC_XML->GetNuisanceParameters());
         if (_verb > 0)
            Info("testCode", "comparing nuisance parameters");
         if (!CompareParameters(*pMC_API->GetNuisanceParameters(), *pMC_XML->GetNuisanceParameters()))
            return false;
      } else
         assert(!pMC_XML->GetNuisanceParameters());

      // compare pdfs
      assert(pMC_API->GetPdf() && pMC_XML->GetPdf());
      RooArgSet pObservables;
      pMC_API->GetObservables()->snapshot(pObservables);
      RooArgSet pGlobalObservables;
      pMC_API->GetGlobalObservables()->snapshot(pGlobalObservables);
      pObservables.addOwned(std::move(pGlobalObservables));

      if (_verb > 0)
         Info("testCode", "comparing PDFs");

      return ComparePDF(*pMC_API->GetPdf(), *pMC_XML->GetPdf(), pObservables, *pWS_API->data("obsData"));
   }

private:
   bool CompareData(const RooAbsData &rData1, const RooAbsData &rData2)
   {
      if (rData1.numEntries() != rData2.numEntries()) {
         Warning("CompareData", "data sets have different numbers of entries: %d vs %d", rData1.numEntries(),
                 rData2.numEntries());
         return false;
      }

      if (rData1.sumEntries() != rData2.sumEntries()) {
         Warning("CompareData", "data sets have different sums of weights");
         return false;
      }

      const RooArgSet *set1 = rData1.get();
      const RooArgSet *set2 = rData2.get();

      if (!CompareParameters(*set1, *set2))
         return false;

      for (auto * par : dynamic_range_cast<RooRealVar *>(*set1)) {
         if (!par)
            continue; // do not test RooCategory
         if (!TMath::AreEqualAbs(rData1.mean(*par), rData2.mean(*par), fTolerance)) {
            Warning("CompareData", "data sets have different means for \"%s\": %.3f vs %.3f", par->GetName(),
                    rData1.mean(*par), rData2.mean(*par));
            return false;
         }

         if (!TMath::AreEqualAbs(rData1.sigma(*par), rData2.sigma(*par), fTolerance)) {
            Warning("CompareData", "data sets have different sigmas for \"%s\": %.3f vs %.3f", par->GetName(),
                    rData1.sigma(*par), rData2.sigma(*par));
            return false;
         }
      }

      return true;
   }

   bool CompareParameters(const RooArgSet &rPars1, const RooArgSet &rPars2, bool bAllowForError = false)
   {
      if (rPars1.size() != rPars2.size()) {
         Warning("CompareParameters", "got different numbers of parameters: %d vs %d", int(rPars1.size()),
                 int(rPars2.size()));
         return false;
      }

      for(auto *arg1 : dynamic_range_cast<RooRealVar *>(rPars1)) {
         // checks only for RooRealVars implemented
         if (!arg1)
            continue;

         RooRealVar *arg2 = (RooRealVar *)rPars2.find(arg1->GetName());

         if (!arg2) {
            Warning("CompareParameters", "did not find observable with name \"%s\"", arg1->GetName());
            return false;
         }

         if (!TMath::AreEqualAbs(arg1->getMin(), arg2->getMin(), fTolerance)) {
            Warning("CompareParameters", "parameters with name \"%s\" have different minima: %.3f vs %.3f",
                    arg1->GetName(), arg1->getMin(), arg2->getMin());
            return false;
         }

         if (!TMath::AreEqualAbs(arg1->getMax(), arg2->getMax(), fTolerance)) {
            Warning("CompareParameters", "parameters with name \"%s\" have different maxima: %.3f vs %.3f",
                    arg1->GetName(), arg1->getMax(), arg2->getMax());
            return false;
         }

         if (arg1->getBins() != arg2->getBins()) {
            Warning("CompareParameters", "parameters with name \"%s\" have different number of bins: %d vs %d",
                    arg1->GetName(), arg1->getBins(), arg2->getBins());
            return false;
         }

         if (arg1->isConstant() != arg2->isConstant()) {
            Warning("CompareParameters", "parameters with name \"%s\" have different constness", arg1->GetName());
            return false;
         }

         if (bAllowForError) {
            if (!TMath::AreEqualAbs(arg1->getVal(), arg2->getVal(),
                                    std::max(fTolerance, 0.1 * std::min(arg1->getError(), arg2->getError())))) {
               Warning("CompareParameters",
                       "parameters with name \"%s\" have different values: %.3f +/- %.3f vs %.3f +/- %.3f",
                       arg1->GetName(), arg1->getVal(), arg1->getError(), arg2->getVal(), arg2->getError());
               return false;
            }
         } else {
            if (!TMath::AreEqualAbs(arg1->getVal(), arg2->getVal(), fTolerance)) {
               Warning("CompareParameters", "parameters with name \"%s\" have different values: %.3f vs %.3f",
                       arg1->GetName(), arg1->getVal(), arg2->getVal());
               return false;
            }

            if (!TMath::AreEqualAbs(arg1->getError(), arg2->getError(), fTolerance)) {
               Warning("CompareParameters", "parameters with name \"%s\" have different errors: %.3f vs %.3f",
                       arg1->GetName(), arg1->getError(), arg2->getError());
               return false;
            }
         }
      }

      return true;
   }

   bool ComparePDF(RooAbsPdf &rPDF1, RooAbsPdf &rPDF2, const RooArgSet &rAllObservables, RooAbsData &rTestData)
   {
      using namespace RooFit;
      using namespace RooStats;

      // options
      const Int_t iSamplingPoints = 100;

      // get variables
      std::unique_ptr<RooArgSet> pVars1{rPDF1.getVariables()};
      std::unique_ptr<RooArgSet> pVars2{rPDF2.getVariables()};

      if (!CompareParameters(*pVars1, *pVars2)) {
         Warning("ComparePDF", "variable sets for PDFs failed check");
         return false;
      }

      std::unique_ptr<RooDataSet> pSamplingPoints{rPDF1.generate(rAllObservables, NumEvents(iSamplingPoints))};
      TH1F *h_diff = new TH1F("h_diff", "relative difference between both PDF;#Delta;Points / 1e-4", 200, -0.01, 0.01);

      float fPDF1value;
      float fPDF2value;
      for (Int_t i = 0; i < pSamplingPoints->numEntries(); ++i) {
         pVars1->assign(*pSamplingPoints->get(i));
         pVars2->assign(*pSamplingPoints->get(i));

         fPDF1value = rPDF1.getVal();
         fPDF2value = rPDF2.getVal();

         float diff = (fPDF1value - fPDF2value);
         if (fPDF1value != 0.f)
            diff /= fPDF1value; // Protect against NaN
         h_diff->Fill(diff);
      }

      bool bResult = true;

      // no deviations > 1%
      if ((h_diff->GetBinContent(0) > 0) || (h_diff->GetBinContent(h_diff->GetNbinsX()) > 0)) {
         Warning("ComparePDF", "PDFs deviate more than 1%% for individual parameter point(s)");
         bResult = false;
      }

      // mean deviation < 0.1%
      if (h_diff->GetMean() > 1e-3) {
         Warning("ComparePDF", "PDFs deviate on average more than 0.1%%");
         bResult = false;
      }

      // clean up
      delete h_diff;

      if (!bResult)
         return false;

      // check fit result to test data
      pVars1->assign(*pVars2);

      // do the fit
      std::string minimizerType = "Minuit2";
      int prec = gErrorIgnoreLevel;
      gErrorIgnoreLevel = kFatal;
      if (gSystem->Load("libMinuit2") < 0)
         minimizerType = "Minuit";
      gErrorIgnoreLevel = prec;

      std::unique_ptr<RooFitResult> r1{
         rPDF1.fitTo(rTestData, Save(), PrintLevel(-1), Minimizer(minimizerType.c_str()))};
      // L.M:  for minuit we need to rest otherwise fit could fail
      if (minimizerType == "Minuit") {
         if (gMinuit) {
            delete gMinuit;
            gMinuit = nullptr;
         }
      }
      std::unique_ptr<RooFitResult> r2{
         rPDF2.fitTo(rTestData, Save(), PrintLevel(-1), Minimizer(minimizerType.c_str()))};

      if (_verb > 0) {
         r1->Print("v");
         r2->Print("v");
      }

      if (!TMath::AreEqualAbs(r1->minNll(), r2->minNll(), 0.05)) {
         Warning("ComparePDF", "likelihood end up in different minima: %.3f vs %.3f", r1->minNll(), r2->minNll());
         return false;
      }

      if (!CompareParameters(*pVars1, *pVars2, true)) {
         Warning("ComparePDF", "variable sets of PDFs differ after fit to test data");
         return false;
      }

      return true;
   }
};

#ifdef HISTFACTORY_XML
TEST(HistFactory, PdfComparison)
#else
TEST(HistFactory, DISABLED_PdfComparison)
#endif
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   PdfComparison test{};
   EXPECT_TRUE(test.testCode());
}
