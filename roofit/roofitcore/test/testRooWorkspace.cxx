// Tests for the RooWorkspace
// Authors: Stephan Hageboeck, CERN  01/2019
//          Jonas Rembser, CERN 05/2025

#include <RooAbsReal.h>
#include <RooArgList.h>
#include <RooGaussian.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooRealVar.h>
#include <RooStats/ModelConfig.h>
#include <RooWorkspace.h>

#include <ROOT/StringUtils.hxx>
#include <TFile.h>
#include <TSystem.h>

#include <gtest/gtest.h>

using namespace RooStats;

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
      ModelConfig modelConfig("ModelConfig", &ws);
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

   ModelConfig *mc = dynamic_cast<ModelConfig *>(w2->genobj("ModelConfig"));
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
      RooStats::ModelConfig modelConfig("ModelConfig", &w);
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
