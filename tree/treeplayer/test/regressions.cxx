#include "TFile.h"
#include "TSystem.h"
#include "TClass.h"
#include "TChain.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TROOT.h"
#include "TH1.h"
#include "TTreeFormula.h"
#include "TString.h"
#include "TLorentzVector.h"
#include <Math/Vector3D.h>
#include <ROOT/TestSupport.hxx>

#include "TTreePlayer.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>
#include <fstream>

// ROOT-10702
TEST(TTreeReaderRegressions, CompositeTypeWithNameClash)
{
   struct Int { int x; };
   gInterpreter->Declare("struct Int { int x; };");

   const auto fname = "ttreereader_compositetypewithnameclash.root";

   {
      TFile f(fname, "recreate");
      Int i{-1};
      int x = 1;
      TTree t("t", "t");
      const auto toJit = "((TTree*)" + std::to_string(reinterpret_cast<std::size_t>(&t)) + ")->Branch(\"i\", (Int*)" +
                         std::to_string(reinterpret_cast<std::size_t>(&i)) + ");";
      gInterpreter->ProcessLine(toJit.c_str());
      t.Branch("x", &x);
      t.Fill();
      t.Write();
      f.Close();
   }

   TFile f(fname);
   TTreeReader r("t", &f);
   TTreeReaderValue<int> iv(r, "i.x");
   TTreeReaderValue<int> xv(r, "x");
   r.Next();
   EXPECT_EQ(xv.GetSetupStatus(), 0);
   if (xv.GetSetupStatus() == 0) {
      EXPECT_EQ(*xv, 1);
   }
   EXPECT_EQ(iv.GetSetupStatus(), 0);
   if (iv.GetSetupStatus() == 0) {
      EXPECT_EQ(*iv, -1);
   }

   gSystem->Unlink(fname);
}

// Regression test for https://github.com/root-project/root/issues/6993
TEST(TTreeReaderRegressions, AutoloadedFriends)
{
   const auto fname = "treereaderautoloadedfriends.root";
   {
      // write a TTree and its friend to the same file:
      // when t1 is read back, it automatically also loads its friend
      TFile f(fname, "recreate");
      TTree t1("t1", "t1");
      TTree t2("t2", "t2");
      int x = 42;
      t2.Branch("x", &x);
      t1.Fill();
      t2.Fill();
      t1.AddFriend(&t2);
      t1.Write();
      t2.Write();
   }

   // reading t2.x via TTreeReader segfaults
   TChain c("t1");
   c.Add(fname);
   c.LoadTree(0);
   TTreeReader r(&c);
   TTreeReaderValue<int> rv(r, "t2.x");
   ASSERT_TRUE(r.Next());
   EXPECT_EQ(*rv, 42);
   EXPECT_FALSE(r.Next());

   gSystem->Unlink(fname);
}

// ROOT-10824
TEST(TTreeReaderRegressions, IndexedFriend)
{
   const auto fname = "treereader_fillindexedfriend.root";

   {
      TFile f(fname, "recreate");
      // Create main tree
      TTree mainTree("mainTree", "mainTree");
      int idx;
      mainTree.Branch("idx", &idx);
      float x;
      mainTree.Branch("x", &x);

      idx = 1;
      x = 1.f;
      mainTree.Fill();
      idx = 1;
      x = 2.f;
      mainTree.Fill();
      idx = 2;
      x = 10.f;
      mainTree.Fill();
      idx = 2;
      x = 20.f;
      mainTree.Fill();
      mainTree.Write();

      // Create aux tree
      TTree auxTree("auxTree", "auxTree");
      auxTree.Branch("idx", &idx);
      std::string s;
      auxTree.Branch("s", &s);
      idx = 1;
      s = "small";
      auxTree.Fill();
      idx = 2;
      s = "big";
      auxTree.Fill();
      auxTree.Write();
      f.Close();
   }

   auto checkTreeReader = [](TTreeReader &r, TTreeReaderValue<float> &rx, TTreeReaderValue<std::string> &rs) {
      ASSERT_TRUE(r.Next());
      EXPECT_EQ(*rx, 1.f);
      EXPECT_EQ(*rs, "small");
      ASSERT_TRUE(r.Next());
      EXPECT_EQ(*rx, 2.f);
      EXPECT_EQ(*rs, "small");
      ASSERT_TRUE(r.Next());
      EXPECT_EQ(*rx, 10.f);
      EXPECT_EQ(*rs, "big");
      ASSERT_TRUE(r.Next());
      EXPECT_EQ(*rx, 20.f);
      EXPECT_EQ(*rs, "big");
      ASSERT_FALSE(r.Next());
   };

   // Test reading back with TTreeReader+TTree
   {
      TFile f(fname);
      auto mainTree = f.Get<TTree>("mainTree");
      auto auxTree = f.Get<TTree>("auxTree");

      auxTree->BuildIndex("idx");
      mainTree->AddFriend(auxTree);

      TTreeReader r(mainTree);
      TTreeReaderValue<float> rx(r, "x");
      TTreeReaderValue<std::string> rs(r, "auxTree.s");
      checkTreeReader(r, rx, rs);
   }

   // Test reading back with TTreeReader+TChain
   {
      TChain mainChain("mainTree", "mainTree");
      mainChain.Add(fname);
      TChain auxChain("auxTree", "auxTree");
      auxChain.Add(fname);

      auxChain.BuildIndex("idx");
      mainChain.AddFriend(&auxChain);

      TTreeReader r(&mainChain);
      TTreeReaderValue<float> rx(r, "x");
      TTreeReaderValue<std::string> rs(r, "auxTree.s");
      checkTreeReader(r, rx, rs);
   }

   gSystem->Unlink(fname);
}

// https://github.com/root-project/root/issues/18066
TEST(TSelectorDrawRegressions, TernaryOperator)
{
   TTree t;
   t.Fill();
   t.Draw("(1?2:3)>>h1(12345,0,20)");
   auto h = gROOT->Get<TH1>("h1");
   ASSERT_EQ(h->GetXaxis()->GetNbins(), 12345); // was ignored before and set to the default 100
   ASSERT_EQ(h->GetBinContent(1235), 1); // FindBin(2) is at 1235
}

// ROOT-4012 (JIRA)
TEST(TTreeFormulaRegressions, ConstantAlias)
{
   TTree t("t", "ti");
   t.SetAlias("w", "3");
   TTreeFormula tf("tf", "4.-w", &t);
   Int_t action;
   TString expr = "w";
   EXPECT_EQ(tf.DefinedVariable(expr, action), 0); // was -1 during the regression
   EXPECT_FLOAT_EQ(tf.EvalInstance(), 1.);
   TTreeFormula tf2("tf2", "4.", &t);
   EXPECT_EQ(tf2.DefinedVariable(expr, action), 0); // was -3 during the regression
   EXPECT_FLOAT_EQ(tf2.EvalInstance(), 4.);
}

// ROOT-8577 (JIRA)
#define MYSTRUCT struct MyS { int x; };
MYSTRUCT
#define TO_LITERAL(string) _QUOTE_(string)
TEST(TTreeFormulaRegressions, WrongName)
{
   gInterpreter->Declare(TO_LITERAL(MYSTRUCT));
   MyS s;
   TLorentzVector v(1, 2, 3, 4);
   TTree t("t", "t");
   t.Branch("s", &s);
   t.Branch("v", &v);
   t.Fill();
   {
      EXPECT_EQ(t.Draw("s.x", ""), 1);
   }
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      // diags.requiredDiag(kError, "TSelectorDraw::AbortProcess", "Variable compilation failed: {s.y,}");
      diags.requiredDiag(kError, "TTreeFormula::ParseWithLeaf", "y is not a datamember of MyS");
      diags.requiredDiag(kError, "TTreeFormula::Compile", " Bad numerical expression : \"s.y\"");
      EXPECT_EQ(t.Draw("s.y", ""), -1);
   }
   {
      EXPECT_EQ(t.Draw("v.Eta()", ""), 1);
   }
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TTreeFormula::ParseWithLeaf", "Unknown method:eta() in TLorentzVector");
      diags.requiredDiag(kError, "TTreeFormula::Compile", " Bad numerical expression : \"v.eta()\"");
      EXPECT_EQ(t.Draw("v.eta()", ""), -1);
   }
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TTreeFormula::ParseWithLeaf", "x is not a datamember of TLorentzVector");
      diags.requiredDiag(kError, "TTreeFormula::Compile", " Bad numerical expression : \"v.x\"");
      EXPECT_EQ(t.Draw("v.x", ""), -1);
   }
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TTreeFormula::ParseWithLeaf", "y is not a datamember of TLorentzVector");
      diags.requiredDiag(kError, "TTreeFormula::Compile", " Bad numerical expression : \"v.y\"");
      EXPECT_EQ(t.Draw("v.y", ""), -1);
   }
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TTreeFormula::ParseWithLeaf", "Unknown method:eta() in MyS");
      diags.requiredDiag(kError, "TTreeFormula::Compile", " Bad numerical expression : \"s.eta()\"");
      EXPECT_EQ(t.Draw("s.eta()", ""), -1);
   }
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kError, "TTreeFormula::ParseWithLeaf", "Unknown method:Eta() in MyS");
      diags.requiredDiag(kError, "TTreeFormula::Compile", " Bad numerical expression : \"s.Eta()\"");
      EXPECT_EQ(t.Draw("s.Eta()", ""), -1);
   }
}

// https://github.com/root-project/root/issues/19814
TEST(TTreeReaderRegressions, UninitializedChain)
{
   auto filename = "eve19814.root";
   auto treename = "events";
   auto brname = "x";
   const int refval = 19814;
   {
      TFile f(filename, "RECREATE");
      TTree t(treename, "");
      int x = refval;
      t.Branch(brname, &x);
      t.Fill();
      f.Write();
   }
   {
      TChain ch(treename);
      ch.Add(filename);
      TTreeReader reader(&ch);
      reader.SetEntriesRange(0, ch.GetEntries());
      EXPECT_EQ(reader.GetEntries(), 1);
      TTreeReaderValue<int> x(reader, brname);
      EXPECT_TRUE(reader.Next());
      EXPECT_EQ(*x, refval);
   }
   gSystem->Unlink(filename);
}

// https://github.com/root-project/root/issues/10423
TEST(TTreeReaderRegressions, XYZVectors)
{
   auto filename1 = "f10423_a.root";
   auto filename2 = "f10423_b.root";
   auto treename = "t";
   for (auto filename : {filename1, filename2}) {
      TFile f(filename, "RECREATE");
      TTree t(treename, treename);
      ROOT::Math::XYZVector x(1, 2, 3);
      std::vector<ROOT::Math::XYZVector> y{ROOT::Math::XYZVector(4, 5, 6)};
      t.Branch("x", &x);
      if (std::string(filename) == std::string(filename1)) {
         // original line:
         t.Branch("y", &y); // commenting this line "fixed" the crash
      } else {
         // Actual trigger:
         auto c = TClass::GetClass("std::vector<ROOT::Math::XYZVector>"); // commenting this line "fixed" the crash
         (void)c;
      }
      t.Fill();
      t.Write();
   }
   for (auto filename : {filename1, filename2}) {
      TFile f(filename, "READ");
      TTreeReader r(treename, &f);
      if (std::string(filename) == std::string(filename1)) {
         TTreeReaderValue<ROOT::Math::XYZVector> rx(r, "x");
         TTreeReaderValue<std::vector<ROOT::Math::XYZVector>> ry(r, "y");
         r.Next();
         EXPECT_EQ(*rx, ROOT::Math::XYZVector(1, 2, 3));
         EXPECT_EQ(*ry, std::vector<ROOT::Math::XYZVector>{ROOT::Math::XYZVector(4, 5, 6)});
      } else {
         TTreeReaderValue<ROOT::Math::XYZVector> rx(r, "x");
         r.Next();
         EXPECT_EQ(*rx, ROOT::Math::XYZVector(1, 2, 3));
      }
   }
   for (auto filename : {filename1, filename2}) {
      gSystem->Unlink(filename);
   }
}

// https://github.com/root-project/root/issues/20226
TEST(TTreeScan, IntOverflow)
{
   struct DatasetRAII {
      const char *fTreeName{"tree_20226"};
      const char *fFileName{"tree_20226.root"};
      DatasetRAII()
      {
         auto file = std::make_unique<TFile>(fFileName, "recreate");
         auto tree = std::make_unique<TTree>(fTreeName, fTreeName);

         int val{};
         tree->Branch("val", &val);
         for (; val < 10; val++)
            tree->Fill();
         file->Write();
      }

      ~DatasetRAII() { std::remove(fFileName); }
   } dataset;

   auto file = std::make_unique<TFile>(dataset.fFileName);
   std::unique_ptr<TTree> tree{file->Get<TTree>(dataset.fTreeName)};

   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(tree->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"tree_20226_regression_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         tree->Scan("val", "", "", TTree::kMaxEntries, 3);

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(************************
*    Row   *       val *
************************
*        3 *         3 *
*        4 *         4 *
*        5 *         5 *
*        6 *         6 *
*        7 *         7 *
*        8 *         8 *
*        9 *         9 *
************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

// https://github.com/root-project/root/issues/20228
TEST(TTreeDraw, IntOverflow)
{
   struct DatasetRAII {
      const char *fTreeName{"tree_20228"};
      const char *fFileName{"tree_20228.root"};
      DatasetRAII()
      {
         auto file = std::make_unique<TFile>(fFileName, "recreate");
         auto tree = std::make_unique<TTree>(fTreeName, fTreeName);

         int val{};
         tree->Branch("val", &val);
         for (; val < 10; val++)
            tree->Fill();
         file->Write();
      }

      ~DatasetRAII() { std::remove(fFileName); }
   } dataset;

   auto file = std::make_unique<TFile>(dataset.fFileName);
   std::unique_ptr<TTree> tree{file->Get<TTree>(dataset.fTreeName)};

   // TTree::Draw returns the number of entries selected. In this case it should be 7,
   // but due to the regression, it was zero
   tree->SetMaxEntryLoop(TTree::kMaxEntries);
   auto nEntriesSelected = tree->Draw("val", "", "", TTree::kMaxEntries, 3);
   EXPECT_EQ(nEntriesSelected, 7);
}

// https://github.com/root-project/root/issues/20248
TEST(TTreeScan, chainNameWithDifferentTreeName)
{
   struct DatasetRAII {
      const char *fTreeName{"tree_20248"};
      const char *fFileName{"tree_20248.root"};
      DatasetRAII()
      {
         auto file = std::make_unique<TFile>(fFileName, "recreate");
         auto tree = std::make_unique<TTree>(fTreeName, fTreeName);

         int val{};
         tree->Branch("val", &val);
         for (; val < 5; val++)
            tree->Fill();
         file->Write();
      }

      ~DatasetRAII() { std::remove(fFileName); }
   } dataset;

   TChain c{"differentNameForChain"};
   c.Add((std::string(dataset.fFileName) + "?#" + dataset.fTreeName).c_str());

   ASSERT_NE(c.FindBranch("differentNameForChain.val"), nullptr);

   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(c.GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"tree_20248_regression_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);
         c.Scan("differentNameForChain.val", "", "colsize=30");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(*********************************************
*    Row   *      differentNameForChain.val *
*********************************************
*        0 *                              0 *
*        1 *                              1 *
*        2 *                              2 *
*        3 *                              3 *
*        4 *                              4 *
*********************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}

// https://github.com/root-project/root/issues/20249
TEST(TTreeScan, TTreeGetBranchOfFriendTChain)
{
   struct DatasetRAII {
      const char *fTreeNameStepZero{"tree_20249_zero"};
      const char *fFileNameStepZero{"tree_20249_zero.root"};
      const char *fTreeNameStepOne{"tree_20249_one"};
      const char *fFileNameStepOne{"tree_20249_one.root"};

      void WriteData(const char *name, const char *treename, int first, int last)
      {
         auto file = std::make_unique<TFile>(name, "RECREATE");
         auto tree = std::make_unique<TTree>(treename, treename);

         int value{};
         tree->Branch("value", &value);

         for (value = first; value < last; ++value) {
            tree->Fill();
         }

         file->Write();
      }

      DatasetRAII()
      {
         WriteData(fFileNameStepZero, fTreeNameStepZero, 3, 7);
         WriteData(fFileNameStepOne, fTreeNameStepOne, 0, 4);
      }

      ~DatasetRAII()
      {
         std::remove(fFileNameStepZero);
         std::remove(fFileNameStepOne);
      }
   } dataset;

   auto chain0 = std::make_unique<TChain>("stepzerochain");
   chain0->Add((std::string(dataset.fFileNameStepZero) + "?#" + dataset.fTreeNameStepZero).c_str());

   auto file1 = std::make_unique<TFile>(dataset.fFileNameStepOne);
   std::unique_ptr<TTree> tree1{file1->Get<TTree>(dataset.fTreeNameStepOne)};
   tree1->AddFriend(chain0.get());

   ASSERT_NE(tree1->FindBranch("stepzerochain.value"), nullptr);

   std::ostringstream strCout;
   {
      if (auto *treePlayer = static_cast<TTreePlayer *>(tree1->GetPlayer())) {
         struct FileRAII {
            const char *fPath;
            FileRAII(const char *name) : fPath(name) {}
            ~FileRAII() { std::remove(fPath); }
         } redirectFile{"tree_20249_regression_redirect.txt"};
         treePlayer->SetScanRedirect(true);
         treePlayer->SetScanFileName(redirectFile.fPath);

         tree1->Scan("value:stepzerochain.value", "", "colsize=24");

         std::ifstream redirectStream(redirectFile.fPath);
         std::stringstream redirectOutput;
         redirectOutput << redirectStream.rdbuf();

         const static std::string expectedScanOut{
            R"Scan(******************************************************************
*    Row   *                    value *      stepzerochain.value *
******************************************************************
*        0 *                        0 *                        3 *
*        1 *                        1 *                        4 *
*        2 *                        2 *                        5 *
*        3 *                        3 *                        6 *
******************************************************************
)Scan"};
         EXPECT_EQ(redirectOutput.str(), expectedScanOut);
      } else
         throw std::runtime_error("Could not retrieve TTreePlayer from main tree!");
   }
}
