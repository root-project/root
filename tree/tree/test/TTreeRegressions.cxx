#include "TMemFile.h"
#include "TLeaf.h"
#include "TChain.h"
#include "TTree.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include "TLeafObject.h"
#include "TH1F.h"
#include "TROOT.h"

#include "gtest/gtest.h"

#include <array>
#include <vector>
#include <memory>

// ROOT-10702
TEST(TTreeRegressions, CompositeTypeWithNameClash)
{
   struct Int {
      int x;
   };
   gInterpreter->Declare("struct Int { int x; };");

   TMemFile f("tree_compositetypewithnameclash.root", "recreate");
   {
      Int i{-1};
      int x = 1;
      TTree t("t", "t");
      const auto toJit = "((TTree*)" + std::to_string(reinterpret_cast<std::size_t>(&t)) + ")->Branch(\"i\", (Int*)" +
                         std::to_string(reinterpret_cast<std::size_t>(&i)) + ");";
      gInterpreter->ProcessLine(toJit.c_str());
      t.Branch("x", &x);
      t.Fill();
      t.Write();
   }

   auto &t = *f.Get<TTree>("t");
   int x = 123;

   const auto ret = t.SetBranchAddress("x", &x);
   EXPECT_EQ(ret, 0);

   t.GetEntry(0);
   EXPECT_EQ(x, 1);

   int ix;
   const auto toJit2 = "((TTree*)" + std::to_string(reinterpret_cast<std::size_t>(&t)) +
                       ")->SetBranchAddress(\"i.x\", (int*)" + std::to_string(reinterpret_cast<std::size_t>(&ix)) +
                       ");";
   gInterpreter->ProcessLine(toJit2.c_str());
   t.GetEntry(0);
   EXPECT_EQ(ix, -1);
}

// ROOT-10942
struct SimpleStruct {
   double a;
   double b;
};

TEST(TTreeRegressions, GetLeafByFullName)
{
   gInterpreter->Declare("struct SimpleStruct { double a; double b; };");
   SimpleStruct c;
   TTree t("t1", "t1");
   t.Branch("c", &c);
   t.Fill();

   EXPECT_TRUE(t.GetLeaf("a") != nullptr);
   EXPECT_TRUE(t.GetLeaf("c.a") != nullptr);
   EXPECT_TRUE(t.GetLeaf("c", "a") != nullptr);
   EXPECT_TRUE(t.GetLeaf(t.GetLeaf("a")->GetFullName()) != nullptr);
}

// Issue #6527
TEST(TTreeRegressions, ChangeFileWithTFileOnStack)
{
   TFile f("ChangeFileWithTFileOnStack.root", "recreate");
   EXPECT_FALSE(f.IsOnHeap());
   TTree t("t", "SetMaxTreeSize(1000)", 99, &f);
   int x;
   auto nentries = 20000;

   t.Branch("x", &x, "x/I");
   t.SetMaxTreeSize(1000);

   for (auto i = 0; i < nentries; i++){
      x = i;
      t.Fill();
   }

   auto cf = t.GetCurrentFile();
   cf->Write();
   cf->Close();

   gSystem->Unlink("ChangeFileWithTFileOnStack.root");
   gSystem->Unlink("ChangeFileWithTFileOnStack_1.root");
   gSystem->Unlink("ChangeFileWithTFileOnStack_2.root");
}

// Issue #6964
TEST(TTreeRegressions, GetLeafAndFriends)
{
   TTree t("t", "t");
   int x = 42;
   std::vector<int> v(1, 42);
   t.Branch("x", &x);
   t.Branch("vec", &v);
   t.Fill();

   TTree t2("t2", "t2");
   t2.Branch("x", &x);
   t2.Branch("vec", &v);
   t2.Fill();

   EXPECT_EQ(t.GetLeaf("asdklj", "x"), nullptr);
   EXPECT_EQ(t.GetLeaf("asdklj", "vec"), nullptr);

   t.AddFriend(&t2);
   EXPECT_EQ(t.GetLeaf("asdklj", "x"), nullptr);
   EXPECT_EQ(t.GetLeaf("asdklj", "vec"), nullptr);
}

// PR #14887
TEST(TTreeRegressions, LeafLongString)
{
   TTree t("t", "t");
   char s[1000];
   memset(s, 'a', 999);
   s[999] = 0;
   t.Branch("s", &s, "s/C");

   s[254] = 0;
   t.Fill();
   s[254] = 'a';
   s[255] = 0;
   t.Fill();
   s[255] = 'a';
   t.Fill();

   s[0] = 0;
   t.GetEntry(0);
   EXPECT_EQ(strlen(s), 254);

   s[0] = 0;
   t.GetEntry(1);
   EXPECT_EQ(strlen(s), 255);

   s[0] = 0;
   t.GetEntry(2);
   EXPECT_EQ(strlen(s), 999);
}

// https://github.com/root-project/root/issues/9319
TEST(TTreeRegressions, PrintClustersRounding)
{
   TMemFile file("tree9319_clusters.root", "RECREATE");
   TTree t("t", "t");
   t.SetAutoFlush(5966);
   int x = 0;
   t.Branch("x", &x);
   for (auto i = 0; i < 10000; ++i) {
      t.Fill();
   }

   testing::internal::CaptureStdout();

   t.Print("clusters");

   const std::string output = testing::internal::GetCapturedStdout();
   #ifdef R__HAS_ZLIB_NG
   // We need this distinction because in some cases, at compression level 1, the heuristics of
   // zlib-ng will opt for not compression the buffer.
   // Here we are interested in clustering anyways
   const auto ref = "******************************************************************************\n"
                    "*Tree    :t         : t                                                      *\n"
                    "*Entries :    10000 : Total =           40973 bytes  File  Size =        120 *\n"
                    "*        :          : Tree compression factor = 199.41                       *\n"
                    "******************************************************************************\n"
                    "Cluster Range #  Entry Start      Last Entry           Size   Number of clusters\n"
                    "0                0                9999                 5966          2\n"
                    "Total number of clusters: 2 \n"; // This was 1 before the fix
   #else
   const auto ref = "******************************************************************************\n"
                    "*Tree    :t         : t                                                      *\n"
                    "*Entries :    10000 : Total =           40973 bytes  File  Size =        202 *\n"
                    "*        :          : Tree compression factor = 118.46                       *\n"
                    "******************************************************************************\n"
                    "Cluster Range #  Entry Start      Last Entry           Size   Number of clusters\n"
                    "0                0                9999                 5966          2\n"
                    "Total number of clusters: 2 \n"; // This was 1 before the fix
   #endif
   EXPECT_EQ(output, ref);
}

// Issue ROOT-9961
TEST(TTreeRegressions, PrintTopOnly)
{
   TTree tree("newtree", "");
   tree.Branch("brancha", 0, "brancha/I");
   tree.Branch("branchb", 0, "branchb/I");

   testing::internal::CaptureStdout();

   tree.Print("toponly");

   const std::string output = testing::internal::GetCapturedStdout();
   const auto ref = "******************************************************************************\n"
                    "*Tree    :newtree   :                                                        *\n"
                    "*Entries :        0 : Total =            1285 bytes  File  Size =          0 *\n"
                    "*        :          : Tree compression factor =   1.00                       *\n"
                    "******************************************************************************\n"
                    "branch: brancha                      0\n"
                    "branch: branchb                      0\n";
   EXPECT_EQ(output, ref);
}

// Issue ROOT-7926
struct Event {
   double x;
   double y;
};
TEST(TTreeRegressions, PrintTopOnlySplit)
{
   gInterpreter->Declare("struct Event { double x; double y; };");
   TTree tree("newtree", "");
   Event ev;
   tree.Branch("ev", &ev); // by default, this calls splitlevel=1
   tree.Fill();

   testing::internal::CaptureStdout();

   tree.Print("toponly");

   const std::string output = testing::internal::GetCapturedStdout();
   const auto ref = "******************************************************************************\n"
                    "*Tree    :newtree   :                                                        *\n"
                    "*Entries :        1 : Total =            2188 bytes  File  Size =          0 *\n"
                    "*        :          : Tree compression factor =   1.00                       *\n"
                    "******************************************************************************\n"
                    "branch: ev                           0\n";
   EXPECT_EQ(output, ref);
}

// https://github.com/root-project/root/issues/12537
TEST(TTreeRegressions, EmptyLeafObject)
{
   TLeafObject tlo;
   EXPECT_EQ(tlo.GetObject(), nullptr);
}

// https://its.cern.ch/jira/browse/ROOT-6741
#define MYSUBCLASS struct MySubClass { int id; double x; };
#define MYCLASS struct MyClass { std::vector<MySubClass> sub; MySubClass *Get(int id) { for (size_t i = 0; i < sub.size(); ++i) if (sub[i].id == id) return &sub[i]; return nullptr; } };
MYSUBCLASS
MYCLASS

TEST(TTreeRegressions, TTreeFormulaMemberIndex)
{
   gInterpreter->Declare(_R_QUOTEVAL_(MYSUBCLASS));
   gInterpreter->Declare(_R_QUOTEVAL_(MYCLASS));

   TTree tree("tree", "tree");
   MyClass mc;
   tree.Branch("mc", &mc);

   MySubClass s;
   s.id = 1;
   s.x = 1.11;
   mc.sub.push_back(s);
   s.id = 23;
   s.x = 2.22;
   mc.sub.push_back(s);
   s.id = -2;
   s.x = 3.33;
   mc.sub.push_back(s);
   tree.Fill();

   Long64_t n1 = tree.Draw("mc.Get(1)->x >> h1", "");
   ASSERT_EQ(n1, 1);
   auto h1 = gROOT->Get<TH1F>("h1");
   ASSERT_FLOAT_EQ(mc.Get(1)->x, h1->GetMean());
   delete h1;

   Long64_t n2 = tree.Draw("mc.Get(23)->x >> h2", "");
   ASSERT_EQ(n2, 1);
   auto h2 = gROOT->Get<TH1F>("h2");
   ASSERT_FLOAT_EQ(mc.Get(23)->x, h2->GetMean());
   delete h2;

   Long64_t n3 = tree.Draw("mc.Get(-2)->x >> h3", "");
   ASSERT_EQ(n3, 1);
   auto h3 = gROOT->Get<TH1F>("h3");
   ASSERT_FLOAT_EQ(mc.Get(-2)->x, h3->GetMean());
   delete h3;
}

// https://its.cern.ch/jira/browse/ROOT-5567
TEST(TTreeRegressions, FindBranchBrackets)
{
   TTree t("t", "");
   UShort_t branch[3];
   t.Branch("branch[3]", branch);
   EXPECT_NE(t.FindBranch("branch[3]"), nullptr);
   EXPECT_EQ(t.FindBranch("branch[3]"), t.GetBranch("branch[3]"));
}

// see https://root-forum.cern.ch/t/bug-or-feature-in-ttree-draw/62862
// Due to a poor binning choice in THLimitsFinder, the histogram didn't contain
// all values.
TEST(TTreeRegressions, DrawAutoBinning)
{
   TTree t;
   Float_t x;
   t.Branch("x", &x);
   x = -999;
   t.Fill();
   x = 0;
   t.Fill();
   t.Draw("x");
   auto h = (TH1 *)gROOT->FindObject("htemp");
   ASSERT_NE(h, nullptr);
   EXPECT_EQ(h->GetEntries(), h->GetEffectiveEntries());
   delete h;
   delete gROOT->FindObject("c1");
}

// Regression for https://github.com/root-project/root/issues/22652
struct RegressionGH22652 : public ::testing::Test {

   constexpr static auto fMainTreeName{"main"};
   constexpr static auto fFriendTreeName{"friend"};
   constexpr static auto fMainTreeFileName{"main_global.root"};
   constexpr static auto fNFiles{3};
   constexpr static auto fNEntriesPerFile{100};
   // file 0: w in [4,5)   file 1: w in [0,1)   file 2: w in [9,10)
   constexpr static std::array<double, 3> fLowerBounds{4.0, 0.0, 9.0};
   constexpr static std::array<const char *, 3> fMainChainFileNames{"main_1.root", "main_2.root", "main_3.root"};
   constexpr static std::array<const char *, 3> fFriendChainFileNames{"friend_1.root", "friend_2.root",
                                                                      "friend_3.root"};

   constexpr static auto fNShortFiles{6};
   constexpr static auto fNEntriesPerShortFile{50};
   constexpr static std::array<const char *, 6> fShortFriendChainFileNames{
      "short_friend_1.root", "short_friend_2.root", "short_friend_3.root",
      "short_friend_4.root", "short_friend_5.root", "short_friend_6.root"};
   // We set the minimum value in the second file to check that the branch address is updated by TChain::GetMinimum
   constexpr static std::array<double, 6> fShortFriendChainValues{10, 0, 30, 40, 50, 60};

   static void SetUpTestSuite()
   {
      {
         // Main TTree with cumulated number of entries
         auto fd = std::make_unique<TFile>(fMainTreeFileName, "RECREATE");
         auto td = std::make_unique<TTree>(fMainTreeName, fMainTreeName);
         double x{};
         td->Branch("x", &x);
         for (const auto &_ : fMainChainFileNames)
            for (int i = 0; i < fNEntriesPerFile; ++i)
               td->Fill();
         fd->Write();
      }

      for (int i = 0; i < fNFiles; i++) {
         // Trees for the main chain
         {
            auto fd = std::make_unique<TFile>(fMainChainFileNames[i], "RECREATE");
            auto td = std::make_unique<TTree>(fMainTreeName, fMainTreeName);
            double x{};
            td->Branch("x", &x);
            for (int j = 0; j < fNEntriesPerFile; j++) {
               // x in [-0.5, 0.495]
               x = j * 0.01 - 0.5;
               td->Fill();
            }
            fd->Write();
         }

         // Trees for the friend chain
         {
            auto ff = std::make_unique<TFile>(fFriendChainFileNames[i], "RECREATE");
            auto tf = std::make_unique<TTree>(fFriendTreeName, fFriendTreeName);
            double w{};
            tf->Branch("w", &w);
            for (int j = 0; j < fNEntriesPerFile; j++) {
               // w in [fLowerBounds[i], fLowerBounds[i] + 1)
               w = fLowerBounds[i] + j * (1.0 / fNEntriesPerFile);
               tf->Fill();
            }
            ff->Write();
         }
      }

      // A second friend chain with the total number of entries but twice as many files (half of the entries per file)
      for (auto i = 0; i < fNShortFiles; i++) {
         auto fd = std::make_unique<TFile>(fShortFriendChainFileNames[i], "RECREATE");
         auto td = std::make_unique<TTree>(fFriendTreeName, fFriendTreeName);
         double w{};
         td->Branch("w", &w);
         for (int j = 0; j < fNEntriesPerShortFile; j++) {
            w = fShortFriendChainValues[i];
            td->Fill();
         }
         fd->Write();
      }
   }

   static void TearDownTestSuite()
   {
      for (const auto &f : fMainChainFileNames)
         std::remove(f);

      for (const auto &f : fFriendChainFileNames)
         std::remove(f);
   }
};

TEST_F(RegressionGH22652, RunMainTChain)
{
   // Main is a TChain, friend is a TChain, entries are aligned
   auto m = std::make_unique<TChain>(fMainTreeName);
   for (const auto &fn : fMainChainFileNames)
      m->Add(fn);

   auto fc = std::make_unique<TChain>(fFriendTreeName);
   for (const auto &fn : fFriendChainFileNames)
      fc->Add(fn);

   m->AddFriend(fc.get());

   EXPECT_DOUBLE_EQ(m->GetMinimum("w"), 0.0);
   EXPECT_DOUBLE_EQ(m->GetMaximum("w"), 9.99);
}

TEST_F(RegressionGH22652, RunMainTTree)
{
   // Main is a TTree, friend is a TChain, entries are aligned

   auto fm = std::make_unique<TFile>(fMainTreeFileName);
   std::unique_ptr<TTree> m{fm->Get<TTree>(fMainTreeName)};

   auto fc = std::make_unique<TChain>(fFriendTreeName);
   for (const auto &fn : fFriendChainFileNames)
      fc->Add(fn);

   m->AddFriend(fc.get());

   EXPECT_DOUBLE_EQ(m->GetMinimum("w"), 0.0);
   EXPECT_DOUBLE_EQ(m->GetMaximum("w"), 9.99);
}

TEST_F(RegressionGH22652, TChainFriendWithShorterFiles)
{
   // Main is a TChain, friend is a TChain, total number of entries is the same
   // but the friend TChain has double the number of files and half the entries
   // per file. This exercises in particular the correct updating of the branch
   // addresses of the friend TChain when it switches to another file even though
   // the main TChain is still traversing the same file.
   auto m = std::make_unique<TChain>(fMainTreeName);
   for (const auto &fn : fMainChainFileNames)
      m->Add(fn);

   auto fc = std::make_unique<TChain>(fFriendTreeName);
   for (const auto &fn : fShortFriendChainFileNames)
      fc->Add(fn);

   m->AddFriend(fc.get());

   EXPECT_DOUBLE_EQ(m->GetMinimum("w"), 0.0);
   EXPECT_DOUBLE_EQ(m->GetMaximum("w"), 60.0);
}