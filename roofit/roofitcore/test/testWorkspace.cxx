/// Tests for the RooWorkspace

#include "RooGlobalFunc.h"
#include "RooGaussian.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooAbsReal.h"
#include "RooStats/ModelConfig.h"

#include "TFile.h"
#include "TSystem.h"

#include "gtest/gtest.h"

using namespace RooStats;

/// ROOT-9777, cloning a RooWorkspace. The ModelConfig did not get updated
/// when a workspace was cloned, and was hence pointing to a non-existing workspace.
///
TEST(RooWorkspace, CloneModelConfig_ROOT_9777)
{
   const char* filename = "ROOT-9777.root";

   RooRealVar x("x", "x", 1, 0, 10);
   RooRealVar mu("mu", "mu", 1, 0, 10);
   RooRealVar sigma("sigma", "sigma", 1, 0, 10);

   RooGaussian pdf("Gauss", "Gauss", x, mu, sigma);

   {
      TFile outfile(filename, "RECREATE");
      
      // now create the model config for this problem
      RooWorkspace* w = new RooWorkspace("ws");
      ModelConfig modelConfig("ModelConfig", w);
      modelConfig.SetPdf(pdf);
      modelConfig.SetParametersOfInterest(RooArgSet(sigma));
      modelConfig.SetGlobalObservables(RooArgSet(mu));
      w->import(modelConfig);

      outfile.WriteObject(w, "ws");
      delete w;
   }
   
   RooWorkspace *w2;
   {
      TFile infile(filename, "READ");
      RooWorkspace *w;
      infile.GetObject("ws", w);
      ASSERT_TRUE(w) << "Workspace not read from file.";

      w2 = new RooWorkspace(*w);
      delete w;
   }
   
   w2->Print();

   ModelConfig *mc = dynamic_cast<ModelConfig*>(w2->genobj("ModelConfig"));
   ASSERT_TRUE(mc) << "ModelConfig not retrieved.";
   mc->Print();

   ASSERT_TRUE(mc->GetGlobalObservables()) << "GlobalObsevables in mc broken.";
   mc->GetGlobalObservables()->Print();

   ASSERT_TRUE(mc->GetParametersOfInterest()) << "ParametersOfInterest in mc broken.";
   mc->GetParametersOfInterest()->Print();

   gSystem->Unlink(filename);
}

