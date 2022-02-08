#include "RooRealVar.h"
#include "RooConstVar.h"
#include "RooWorkspace.h"

#include "TROOT.h"

#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include "gtest/gtest.h"

#include "RooGlobalFunc.h"

// includes for RooArgusBG
#include "RooArgusBG.h"
#include "RooGaussian.h"
#include "RooAddPdf.h"

// includes for SimultaneousGaussians
#include "RooRealVar.h"
#include "RooSimultaneous.h"
#include "RooProdPdf.h"
#include "RooCategory.h"


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

   auto etcDir = std::string(TROOT::GetEtcDir());
   RooJSONFactoryWSTool::loadExportKeys(etcDir + "/RooFitHS3_wsexportkeys.json");
   RooJSONFactoryWSTool::loadFactoryExpressions(etcDir + "/RooFitHS3_wsfactoryexpressions.json");

   RooWorkspace work;
   work.import(model);
   RooJSONFactoryWSTool tool(work);
   tool.exportJSON("argus.json");
};

TEST(RooFitHS3, SimultaneousGaussians)
{
   using namespace RooFit;

   // Import keys and factory expressions files for the RooJSONFactoryWSTool.
   auto etcDir = std::string(TROOT::GetEtcDir());
   RooJSONFactoryWSTool::loadExportKeys(etcDir + "/RooFitHS3_wsexportkeys.json");
   RooJSONFactoryWSTool::loadFactoryExpressions(etcDir + "/RooFitHS3_wsfactoryexpressions.json");

   // Create a test model: RooSimultaneous with Gaussian in one component, and
   // product of two Gaussians in the other.
   RooRealVar x("x", "x", -8, 8);
   RooRealVar mean("mean", "mean", 0, -8, 8);
   RooRealVar sigma("sigma", "sigma", 0.3, 0.1, 10);
   RooGaussian g1("g1", "g1", x, mean, sigma);
   RooGaussian g2("g2", "g2", x, mean, RooConst(0.3));
   RooProdPdf model("model", "model", RooArgList{g1, g2});
   RooGaussian model_ctl("model_ctl", "model_ctl", x, mean, sigma);
   RooCategory sample("sample", "sample", {{"physics", 0}, {"control", 1}});
   RooSimultaneous simPdf("simPdf", "simultaneous pdf", sample);
   simPdf.addPdf(model, "physics");
   simPdf.addPdf(model_ctl, "control");

   // this is a handy way of triggering the creation of a ModelConfig upon re-import
   simPdf.setAttribute("toplevel");

   // Export to JSON
   {
      RooWorkspace ws{"workspace"};
      ws.import(simPdf);
      RooJSONFactoryWSTool tool{ws};
      tool.exportJSON("simPdf.json");
      // Output can be pretty-printed with `python -m json.tool simPdf.json`
   }

   // Import JSON
   {
      RooWorkspace ws{"workspace"};
      RooJSONFactoryWSTool tool{ws};
      tool.importJSON("simPdf.json");

      ASSERT_TRUE(ws.pdf("g1"));
      ASSERT_TRUE(ws.pdf("g2"));
   }
}
