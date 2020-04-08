#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooArgusBG.h"
#include "RooGaussian.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooWorkspace.h"

#include "TSystem.h"

#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include "gtest/gtest.h"

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif

using namespace RooFit;

TEST(RooFitHS3, RooArgusBG)
{
   auto &msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   // --- Observable ---
   RooRealVar mes("mes", "m_{ES} (GeV)", 5.20, 5.30);

   // --- Parameters ---
   RooRealVar sigmean("sigmean", "B^{#pm} mass", 5.28, 5.20, 5.30);
   RooRealVar sigwidth("sigwidth", "B^{#pm} width", 0.0027, 0.001, 1.);

   // --- Build Gaussian PDF ---
   RooGaussian signalModel("signal", "signal PDF", mes, sigmean, sigwidth);

   // --- Build Argus background PDF ---
   RooRealVar argpar("argpar", "argus shape parameter", -20.0, -100., -1.);
   RooArgusBG background("background", "Argus PDF", mes, RooConst(5.291), argpar);

   // --- Construct signal+background PDF ---
   RooRealVar nsig("nsig", "#signal events", 200, 0., 10000);
   RooRealVar nbkg("nbkg", "#background events", 800, 0., 10000);
   RooAddPdf model("model", "g+a", RooArgList(signalModel, background), RooArgList(nsig, nbkg));

   std::string rootetcPath = gSystem->Getenv("ROOTSYS");
   RooJSONFactoryWSTool::loadExportKeys(rootetcPath + "/etc/root/RooFitHS3_wsexportkeys.json");
   RooWorkspace work;
   work.import(model);
   RooJSONFactoryWSTool tool(work);
   tool.exportJSON("argus.json");
};
