// Tests for the RooWorkspace
// Authors: Stephan Hageboeck, CERN  01/2019
//          Jonas Rembser, CERN 05/2025

#include <RooAbsReal.h>
#include <RooAddPdf.h>
#include <RooArgList.h>
#include <RooBreitWigner.h>
#include <RooConstVar.h>
#include <RooExponential.h>
#include <RooFFTConvPdf.h>
#include <RooFit/ModelConfig.h>
#include <RooGaussian.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooPlot.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <ROOT/StringUtils.hxx>
#include <TFile.h>
#include <TSystem.h>

#include <gtest/gtest.h>

/// ROOT-9777, cloning a RooWorkspace. The ModelConfig did not get updated
/// when a workspace was cloned, and was hence pointing to a non-existing workspace.
///
TEST(RooWorkspace, CloneModelConfig_ROOT_9777)
{
   constexpr bool verbose = false;

   const char *filename = "ROOT-9777.root";

   RooRealVar x("x", "x", 1, 0, 10);
   RooRealVar mu("mu", "mu", 1, 0, 10);
   RooRealVar sigma("sigma", "sigma", 1, 0.01, 10);

   RooGaussian pdf("Gauss", "Gauss", x, mu, sigma);

   {
      TFile outfile(filename, "RECREATE");

      // now create the model config for this problem
      RooWorkspace ws{"ws"};
      RooFit::ModelConfig modelConfig("ModelConfig", &ws);
      modelConfig.SetPdf(pdf);
      modelConfig.SetParametersOfInterest(RooArgSet(sigma));
      modelConfig.SetGlobalObservables(RooArgSet(mu));
      ws.import(modelConfig);

      outfile.WriteObject(&ws, "ws");
   }

   RooWorkspace *w2;
   {
      TFile infile(filename, "READ");
      std::unique_ptr<RooWorkspace> ws{infile.Get<RooWorkspace>("ws")};
      ASSERT_TRUE(ws) << "Workspace not read from file.";

      w2 = new RooWorkspace(*ws);
   }

   if (verbose)
      w2->Print();

   auto *mc = dynamic_cast<RooFit::ModelConfig *>(w2->genobj("ModelConfig"));
   ASSERT_TRUE(mc) << "ModelConfig not retrieved.";
   mc->Print();

   ASSERT_TRUE(mc->GetGlobalObservables()) << "GlobalObsevables in mc broken.";
   if (verbose)
      mc->GetGlobalObservables()->Print();

   ASSERT_TRUE(mc->GetParametersOfInterest()) << "ParametersOfInterest in mc broken.";
   if (verbose)
      mc->GetParametersOfInterest()->Print();

   gSystem->Unlink(filename);
}

/// Set up a simple workspace for later tests.
class TestRooWorkspaceWithGaussian : public ::testing::Test {
protected:
   TestRooWorkspaceWithGaussian()
   {
      RooRealVar x("x", "x", 1, 0, 10);
      RooRealVar mu("mu", "mu", 1, 0, 10);
      RooRealVar sigma("sigma", "sigma", 1, 0.01, 10);

      RooGaussian pdf("Gauss", "Gauss", x, mu, sigma);

      TFile outfile(_filename, "RECREATE");

      // now create the model config for this problem
      RooWorkspace w("ws");
      RooFit::ModelConfig modelConfig("ModelConfig", &w);
      modelConfig.SetPdf(pdf);
      modelConfig.SetParametersOfInterest(RooArgSet(sigma));
      modelConfig.SetGlobalObservables(RooArgSet(mu));
      w.import(modelConfig);

      outfile.WriteObject(&w, "ws");
   }

   ~TestRooWorkspaceWithGaussian() override { gSystem->Unlink(_filename); }

   const char *_filename = "ROOT-9777.root";
};

/// Test the string tokeniser that does all the string splitting for the RooWorkspace
/// implementation.
TEST(RooHelpers, Tokeniser)
{
   const bool skipEmpty = true;

   std::vector<std::string> tok = ROOT::Split("abc, def, ghi", ", ", skipEmpty);
   EXPECT_EQ(tok.size(), 3U);
   EXPECT_EQ(tok[0], "abc");
   EXPECT_EQ(tok[1], "def");
   EXPECT_EQ(tok[2], "ghi");

   std::vector<std::string> tok2 = ROOT::Split("abc, def", ":", skipEmpty);
   EXPECT_EQ(tok2.size(), 1U);
   EXPECT_EQ(tok2[0], "abc, def");

   std::vector<std::string> tok3 = ROOT::Split(",  ,abc, def,", ", ", skipEmpty);
   EXPECT_EQ(tok3.size(), 2U);
   EXPECT_EQ(tok3[0], "abc");
   EXPECT_EQ(tok3[1], "def");

   std::vector<std::string> tok4 = ROOT::Split(",  ,abc, def,", ",", skipEmpty);
   EXPECT_EQ(tok4.size(), 3U);
   EXPECT_EQ(tok4[0], "  ");
   EXPECT_EQ(tok4[1], "abc");
   EXPECT_EQ(tok4[2], " def");
}

/// Test proper string handling when importing an object from a workspace
/// in a different file.
TEST_F(TestRooWorkspaceWithGaussian, ImportFromFile)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::ObjectHandling, true};

   std::ostringstream spec;
   spec << _filename << ":" << "ws:Gauss";

   RooWorkspace w("ws");

   // Expect successful import:
   EXPECT_FALSE(w.import(spec.str().c_str()));

#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
   // Expect import failures:
   RooHelpers::HijackMessageStream hijack(RooFit::ERROR, RooFit::InputArguments, "ws");
   EXPECT_TRUE(w.import("bogus:abc"));
   EXPECT_FALSE(hijack.str().empty());

   hijack.stream().str("");
   ASSERT_TRUE(hijack.str().empty());
   EXPECT_TRUE(w.import((spec.str() + "bogus").c_str()));
   EXPECT_FALSE(hijack.str().empty());
#endif
}

/// [ROOT-7921] When using EDIT, cannot build PDFs from edit PDF.
TEST_F(TestRooWorkspaceWithGaussian, RooCustomiserInterface)
{
   TFile file(_filename, "READ");
   RooWorkspace *ws;
   file.GetObject("ws", ws);
   ASSERT_NE(ws, nullptr);

   // Prepare
   ASSERT_NE(ws->factory("SUM:sum(a[0.5,0,1]*Gauss,Gauss)"), nullptr);
   ASSERT_NE(ws->factory("expr:sig2(\"1 + @0 * @1\", {sigma_alpha[0.1], theta_alpha[0, -5, 5]})"), nullptr);
   ASSERT_NE(ws->factory("EDIT::editPdf(sum, sigma=sig2)"), nullptr);
   ASSERT_NE(ws->factory("Gaussian::constraint_alpha(global_alpha[0], theta_alpha, 1)"), nullptr);

   // Build a product using the edited pdf. This failed because of ROOT-7921
   // Problem was in RooCustomizer::CustIFace::create
   EXPECT_NE(ws->factory("PROD::model_constrained(editPdf, constraint_alpha)"), nullptr);

   // Test the other code path in RooCustomizer::CustIFace::create.
   // Edit the top-level pdf in-place, replacing all existing conflicting nodes in the workspace by <node>_orig
   ASSERT_NE(ws->factory("EDIT::model_constrained(model_constrained, mu=mu2[-1,-10,10])"), nullptr);

   // Test that the new model_constrained has been altered
   auto model_constrained = ws->pdf("model_constrained");
   ASSERT_NE(model_constrained, nullptr);
   EXPECT_TRUE(model_constrained->dependsOn(*ws->var("mu2")));
   EXPECT_FALSE(model_constrained->dependsOn(*ws->var("mu")));

   // Test that the old model still exists suffixed with _orig
   auto model_constrained_orig = ws->pdf("model_constrained_orig");
   ASSERT_NE(model_constrained_orig, nullptr);
   EXPECT_TRUE(model_constrained_orig->dependsOn(*ws->var("mu")));
   EXPECT_FALSE(model_constrained_orig->dependsOn(*ws->var("mu2")));
   EXPECT_NE(ws->pdf("Gauss_editPdf_orig"), nullptr);
}

/// Test that things still work when hash lookup for elements
/// is performed.
TEST_F(TestRooWorkspaceWithGaussian, HashLookupInWorkspace)
{
   TFile file(_filename, "READ");
   RooWorkspace *ws;
   file.GetObject("ws", ws);
   ASSERT_NE(ws, nullptr);

   ws->useFindsWithHashLookup(true);

   // Prepare
   ASSERT_NE(ws->factory("SUM:sum(a[0.5,0,1]*Gauss,Gauss)"), nullptr);
   ASSERT_NE(ws->factory("expr:sig2(\"1 + @0 * @1\", {sigma_alpha[0.1], theta_alpha[0, -5, 5]})"), nullptr);
   ASSERT_NE(ws->factory("EDIT::editPdf(sum, sigma=sig2)"), nullptr);
   ASSERT_NE(ws->factory("Gaussian::constraint_alpha(global_alpha[0], theta_alpha, 1)"), nullptr);

   // Build a product using the edited pdf. This failed because of ROOT-7921
   // Problem was in RooCustomizer::CustIFace::create
   EXPECT_NE(ws->factory("PROD::model_constrained(editPdf, constraint_alpha)"), nullptr);

   // Test the other code path in RooCustomizer::CustIFace::create.
   // Edit the top-level pdf in-place, replacing all existing conflicting nodes in the workspace by <node>_orig
   ASSERT_NE(ws->factory("EDIT::model_constrained(model_constrained, mu=mu2[-1,-10,10])"), nullptr);

   // Test that the new model_constrained has been altered
   auto model_constrained = ws->pdf("model_constrained");
   ASSERT_NE(model_constrained, nullptr);
   EXPECT_TRUE(model_constrained->dependsOn(*ws->var("mu2")));
   EXPECT_FALSE(model_constrained->dependsOn(*ws->var("mu")));

   // Test that the old model still exists suffixed with _orig
   auto model_constrained_orig = ws->pdf("model_constrained_orig");
   ASSERT_NE(model_constrained_orig, nullptr);
   EXPECT_TRUE(model_constrained_orig->dependsOn(*ws->var("mu")));
   EXPECT_FALSE(model_constrained_orig->dependsOn(*ws->var("mu2")));
   EXPECT_NE(ws->pdf("Gauss_editPdf_orig"), nullptr);
}

/// Covers an issue about a RooAddPdf constructor not properly picked up by
/// RooFactoryWSTool.
TEST(RooWorkspace, Issue_7965)
{
   RooWorkspace ws{"ws"};
   ws.factory("RooAddPdf::addPdf({})");

   ASSERT_NE(ws.pdf("addPdf"), nullptr);
}

/// Covers an issue about the RooProdPdf constructor taking a RooFit collection
/// not working and the RooProduct constructors behaving inconsistently.
TEST(RooWorkspace, Issue_7809)
{
   RooWorkspace ws;
   ws.factory("RooGaussian::a(x[-10,10],0.,1.)");
   ws.factory("RooGaussian::b(y[-10,10],0.,1.)");

   ws.factory("RooProdPdf::p1({a,b})");
   ws.factory("RooProduct::p2({x,y})");

   ws.factory("RooProdPdf::p3(a,b)");
   ws.factory("RooProduct::p4(x,y)");

   ASSERT_EQ(static_cast<RooProdPdf *>(ws.pdf("p1"))->pdfList().size(), 2);
   ASSERT_EQ(static_cast<RooProduct *>(ws.function("p2"))->components().size(), 2);
   ASSERT_EQ(static_cast<RooProdPdf *>(ws.pdf("p3"))->pdfList().size(), 2);
   ASSERT_EQ(static_cast<RooProduct *>(ws.function("p4"))->components().size(), 2);
}

/// Check if handles to the owning RooWorkspace are correctly updated when
/// copying the workspace.
TEST(RooWorkspace, RooWorkspaceHandleCopy)
{
   RooWorkspace ws1{"ws"};
   RooFit::ModelConfig mc("ModelConfig");
   mc.SetWS(ws1);
   ws1.import(mc);
   auto mc1 = static_cast<RooFit::ModelConfig *>(ws1.obj("ModelConfig"));
   EXPECT_EQ(mc1->GetWS(), &ws1);

   RooWorkspace ws2{ws1};
   auto mc2 = static_cast<RooFit::ModelConfig *>(ws2.obj("ModelConfig"));
   EXPECT_EQ(mc2->GetWS(), &ws2);
}

/// Check if handles to the owning RooWorkspace are correctly updated when
/// streaming the workspace.
TEST(RooWorkspace, RooWorkspaceHandleCopyWithStreamer)
{
   {
      RooWorkspace ws1{"ws"};
      RooFit::ModelConfig mc("ModelConfig");
      mc.SetWS(ws1);
      ws1.import(mc);
      auto mc1 = static_cast<RooFit::ModelConfig *>(ws1.obj("ModelConfig"));
      EXPECT_EQ(mc1->GetWS(), &ws1);

      ws1.writeToFile("test_rooworkspace.root");
   }

   std::unique_ptr<TFile> file(TFile::Open("test_rooworkspace.root", "read"));
   auto &ws2 = *file->Get<RooWorkspace>("ws");

   auto mc2 = static_cast<RooFit::ModelConfig *>(ws2.obj("ModelConfig"));
   EXPECT_EQ(mc2->GetWS(), &ws2);
}

/// Like the RooWorkspaceHandleCopyWithStreamer test, but with a workspace that
/// contains the old ModelConfig class version 6, where the reference to the
/// owning workspace was still a TRef and not a transient raw pointer. So this
/// test checks if the schema evolution works correctly.
TEST(RooWorkspace, RooWorkspaceHandleCopyWithStreamerFromModelConfig6)
{
   std::unique_ptr<TFile> file(TFile::Open("workspace_with_model_config_classdef_6.root", "read"));
   auto &ws2 = *file->Get<RooWorkspace>("workspace");

   auto mc2 = static_cast<RooFit::ModelConfig *>(ws2.obj("ModelConfig"));
   EXPECT_EQ(mc2->GetWS(), &ws2);
}

// This test covers an issue that was reported after updates to ROOT IO:
//
//     https://github.com/root-project/root/issues/10282
//
// The reproducer workspace was created with ROOT 6.26.00 with the following
// script:
//
// ```C++
// auto f = TFile::Open("toyws/WS-boostedHbb-glob_xs_toy.root");
//
// RooWorkspace ws{"combWS", "__temp__"};
// ws.defineSet("myset", RooArgSet{});
// ws.writeToFile("test_workspace_01.root");
// ```
//
// The original toy workspace was created by ATLAS users and it can be found
// here (some ROOT 6.24 release was used to produce these workspaces):
//
//     https://gitlab.cern.ch/kran/toyws/-/tree/master
//
// The script above aimed for a workspace that is as tiny as possible while
// reproducing the problem. These three conditions were found sufficient to
// create a workspace that is affected by issue 10282:
//
//   1. A file with a broken workspace needs to be opened
//   2. The reproducer workspace needs to have the same name as the broken
//      workspace in that file
//   3. The reproducer workspace must have some RooArgSet defined
TEST(RooWorkspace, Issue_10282)
{
   auto f = TFile::Open("test_workspace_01.root");
   auto *ws = f->Get<RooWorkspace>("combWS");

   ASSERT_NE(ws->set("myset"), nullptr);
   ASSERT_EQ(ws->set("myset")->size(), 0);
}

void createWorkspaceForIssue10577(RooWorkspace &ws, const double delta = 0)
{
   const double xmin = 986;
   const double xmax = 1090;
   const double normMin = xmin + delta;

   RooRealVar x("x", "x", xmin, xmax);
   // range in which the normalizations (integrals) are given
   x.setRange("norm", normMin, xmax);
   // to make RooFFTConvPdf provide values in broadest used range
   x.setRange("cache", std::min(normMin, xmin), xmax);
   ws.import(x);

   RooRealVar width("gamma", "gamma", 4.266, "MeV/c^{2}");
   RooRealVar mean("mean", "mean", 1019.461, 1015.0, 1025.0, "MeV/c^{2}");
   RooRealVar sigma("sigma", "sigma", 1.0, 0.05, 2.5, "MeV/c^{2}");

   RooGaussian det("det", "det", x, RooFit::RooConst(0), sigma);
   RooBreitWigner bw("bw", "bw", x, mean, width);

   x.setBins(10000, "cache"); // for FFT sampling
   RooFFTConvPdf signal("signal", "signal", x, bw, det);
   ws.import(signal, RooFit::RecycleConflictNodes());
}

// Reproducer from https://github.com/root-project/root/issues/10577
TEST(RooWorkspace, Issue_10577)
{
   auto doPlot = [] {
      RooWorkspace ws("workspace");
      createWorkspaceForIssue10577(ws);

      std::unique_ptr<RooPlot> frame{ws.var("x")->frame()};
      ws.pdf("signal")->plotOn(frame.get());
   };

   auto doIntegral = [] {
      RooWorkspace ws("workspace");
      createWorkspaceForIssue10577(ws, -6);

      RooRealVar &x = *ws.var("x");
      std::unique_ptr<RooAbsReal> integralObject{
         ws.pdf("signal")->createIntegral(x, RooFit::NormSet(x), RooFit::Range("norm"))};
      const double integral = integralObject->getVal();
      return integral;
   };

   const double expected = doIntegral();
   doPlot();
   const double afterPlot = doIntegral();

   EXPECT_DOUBLE_EQ(afterPlot, expected);
}

namespace {

/// Small composition model used by the import tests below.
struct ImportTestModel {
   RooRealVar x{"x", "x", 0, 10};
   RooRealVar mean{"mean", "mean", 5, 0, 10};
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 5.0};
   RooRealVar c{"c", "c", -0.1, -1.0, 0.0};
   RooRealVar f{"f", "f", 0.4, 0.0, 1.0};
   RooGaussian gauss{"gauss", "gaussian", x, mean, sigma};
   RooExponential expo{"expo", "exponential", x, c};
   RooAddPdf model{"model", "model", RooArgList{gauss, expo}, f};
};

bool hasServer(RooAbsArg const &node, RooAbsArg const &server)
{
   for (RooAbsArg *s : node.servers()) {
      if (s == &server) {
         return true;
      }
   }
   return false;
}

/// The imported graph must be fully self-contained: every server of every node
/// in the workspace has to be owned by that same workspace. If the renaming
/// bookkeeping in import() is wrong, nodes end up still pointing at the
/// original objects outside the workspace.
void expectSelfContained(RooWorkspace &ws)
{
   for (RooAbsArg *node : ws.components()) {
      EXPECT_EQ(node->workspace(), &ws) << "node " << node->GetName() << " is not owned by the workspace";
      for (RooAbsArg *server : node->servers()) {
         EXPECT_TRUE(ws.components().containsInstance(*server))
            << "server " << server->GetName() << " of " << node->GetName() << " is not owned by the workspace";
      }
   }
}

} // namespace

/// Importing a computation graph clones it, and clones it a second time to make
/// any renaming effective. Check that each renaming mode results in a correctly
/// named and correctly wired workspace, and that the values are preserved.
TEST(RooWorkspace, ImportRenamingModes)
{
   // The pdfs are deliberately evaluated without a normalization set, which is
   // fine here because only the imported and the original value are compared.
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::ERROR};

   // No renaming at all
   {
      ImportTestModel m;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m.model, RooFit::Silence());

      ASSERT_NE(ws.pdf("model"), nullptr);
      EXPECT_DOUBLE_EQ(ws.pdf("model")->getVal(), m.model.getVal());
      expectSelfContained(ws);
   }

   // Name conflict resolved by renaming the incoming nodes
   {
      ImportTestModel m1;
      ImportTestModel m2;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m1.model, RooFit::Silence());
      ws.import(m2.model, RooFit::RenameConflictNodes("v2"), RooFit::Silence());

      ASSERT_NE(ws.pdf("model_v2"), nullptr);
      EXPECT_NE(ws.pdf("gauss_v2"), nullptr);
      EXPECT_NE(ws.pdf("expo_v2"), nullptr);
      // The renamed top node must be wired to the renamed components
      EXPECT_TRUE(hasServer(*ws.pdf("model_v2"), *ws.pdf("gauss_v2")));
      EXPECT_TRUE(hasServer(*ws.pdf("model_v2"), *ws.pdf("expo_v2")));
      EXPECT_STREQ(ws.pdf("model_v2")->getStringAttribute("origName"), "model");
      EXPECT_DOUBLE_EQ(ws.pdf("model_v2")->getVal(), m2.model.getVal());
      expectSelfContained(ws);
   }

   // Name conflict resolved by renaming the nodes already in the workspace
   {
      ImportTestModel m1;
      ImportTestModel m2;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m1.model, RooFit::Silence());
      ws.import(m2.model, RooFit::RenameConflictNodes("old", true), RooFit::Silence());

      ASSERT_NE(ws.pdf("model"), nullptr);
      EXPECT_NE(ws.pdf("model_old"), nullptr);
      EXPECT_DOUBLE_EQ(ws.pdf("model")->getVal(), m2.model.getVal());
      expectSelfContained(ws);
   }

   // Rename every node, not just the conflicting ones
   {
      ImportTestModel m1;
      ImportTestModel m2;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m1.model, RooFit::Silence());
      ws.import(m2.model, RooFit::RenameAllNodes("all"), RooFit::Silence());

      ASSERT_NE(ws.pdf("model_all"), nullptr);
      EXPECT_TRUE(hasServer(*ws.pdf("model_all"), *ws.pdf("gauss_all")));
      EXPECT_DOUBLE_EQ(ws.pdf("model_all")->getVal(), m2.model.getVal());
      expectSelfContained(ws);
   }

   // Rename a single variable
   {
      ImportTestModel m;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m.model, RooFit::RenameVariable("x", "obs"), RooFit::Silence());

      ASSERT_NE(ws.var("obs"), nullptr);
      EXPECT_EQ(ws.var("x"), nullptr);
      EXPECT_TRUE(hasServer(*ws.pdf("gauss"), *ws.var("obs")));
      EXPECT_DOUBLE_EQ(ws.pdf("model")->getVal(), m.model.getVal());
      expectSelfContained(ws);
   }

   // Rename all variables at once
   {
      ImportTestModel m;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m.model, RooFit::RenameAllVariables("sfx"), RooFit::Silence());

      ASSERT_NE(ws.var("x_sfx"), nullptr);
      EXPECT_EQ(ws.var("x"), nullptr);
      EXPECT_NE(ws.var("mean_sfx"), nullptr);
      EXPECT_TRUE(hasServer(*ws.pdf("gauss"), *ws.var("x_sfx")));
      EXPECT_DOUBLE_EQ(ws.pdf("model")->getVal(), m.model.getVal());
      expectSelfContained(ws);
   }
}

/// Importing with RecycleConflictNodes() has to connect the imported nodes to
/// the same-name nodes already in the workspace instead of duplicating them.
/// This mode is used heavily when building workspaces incrementally, like in
/// HistFactory, so it takes a shortcut that skips cloning the already-imported
/// parts of the computation graph. Check that the resulting workspace is
/// correctly wired in the scenarios that shortcut has to handle.
TEST(RooWorkspace, ImportRecycleConflictNodes)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::ERROR};

   // Incremental build: import components first, then a top-level pdf reusing
   // them. Only the top-level node is new.
   {
      ImportTestModel m;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m.gauss, RooFit::Silence());
      ws.import(m.expo, RooFit::Silence());
      ws.import(m.model, RooFit::RecycleConflictNodes(), RooFit::Silence());

      ASSERT_NE(ws.pdf("model"), nullptr);
      EXPECT_TRUE(hasServer(*ws.pdf("model"), *ws.pdf("gauss")));
      EXPECT_TRUE(hasServer(*ws.pdf("model"), *ws.pdf("expo")));
      EXPECT_DOUBLE_EQ(ws.pdf("model")->getVal(), m.model.getVal());
      expectSelfContained(ws);
   }

   // Re-importing an already existing graph must be a no-op.
   {
      ImportTestModel m;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m.model, RooFit::Silence());
      RooAbsPdf *modelBefore = ws.pdf("model");
      const std::size_t nComponents = ws.components().size();
      EXPECT_FALSE(ws.import(m.model, RooFit::RecycleConflictNodes(), RooFit::Silence()));

      EXPECT_EQ(ws.pdf("model"), modelBefore);
      EXPECT_EQ(ws.components().size(), nComponents);
      expectSelfContained(ws);
   }

   // The values of recycled nodes must come from the workspace copies, not
   // from the incoming objects.
   {
      ImportTestModel m;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m.gauss, RooFit::Silence());
      m.mean.setVal(7.); // changed after the import: the workspace copy stays at 5
      ws.import(m.model, RooFit::RecycleConflictNodes(), RooFit::Silence());

      EXPECT_DOUBLE_EQ(ws.var("mean")->getVal(), 5.);
      expectSelfContained(ws);
   }

   // When a same-name node conflicts, the workspace copy wins and keeps its
   // own structure, but new nodes below the conflicting node are still
   // imported (as unreferenced nodes), like in the general import code path.
   {
      RooWorkspace ws{"ws", "ws"};
      {
         RooRealVar x{"x", "x", 0., 10.};
         RooRealVar mean{"mean", "mean", 5., 0., 10.};
         RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 5.0};
         RooGaussian sub{"sub", "sub", x, mean, sigma};
         ws.import(sub, RooFit::Silence());
      }
      RooRealVar x{"x", "x", 0., 10.};
      RooRealVar theta{"theta", "theta", -0.1, -1.0, 0.0};
      RooExponential sub{"sub", "sub", x, theta}; // same name, different structure
      RooRealVar f{"f", "f", 0.4, 0.0, 1.0};
      RooGaussian other{"other", "other", x, 1.0, 2.0};
      RooAddPdf top{"top", "top", RooArgList{sub, other}, f};
      ws.import(top, RooFit::RecycleConflictNodes(), RooFit::Silence());

      // The workspace copy of "sub" keeps its structure ...
      EXPECT_NE(ws.var("mean"), nullptr);
      EXPECT_TRUE(hasServer(*ws.pdf("top"), *ws.pdf("sub")));
      EXPECT_TRUE(hasServer(*ws.pdf("sub"), *ws.var("mean")));
      // ... and the new parameter below the conflicting node is imported
      EXPECT_NE(ws.var("theta"), nullptr);
      expectSelfContained(ws);
   }

   // A deeper boundary: the top-level pdf of the previous import becomes an
   // intermediate node of the newly imported graph.
   {
      ImportTestModel m;
      RooWorkspace ws{"ws", "ws"};
      ws.import(m.model, RooFit::Silence());
      RooRealVar y{"y", "y", 1.0, 0.0, 10.0};
      RooGaussian gaussy{"gaussy", "gaussy", y, m.mean, m.sigma};
      RooProdPdf prod{"prod", "prod", RooArgList{m.model, gaussy}};
      ws.import(prod, RooFit::RecycleConflictNodes(), RooFit::Silence());

      ASSERT_NE(ws.pdf("prod"), nullptr);
      EXPECT_TRUE(hasServer(*ws.pdf("prod"), *ws.pdf("model")));
      ASSERT_NE(ws.pdf("gaussy"), nullptr);
      EXPECT_TRUE(hasServer(*ws.pdf("gaussy"), *ws.var("mean")));
      expectSelfContained(ws);
   }
}
