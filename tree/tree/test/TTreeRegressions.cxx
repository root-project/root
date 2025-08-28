#include "TMemFile.h"
#include "TLeaf.h"
#include "TTree.h"
#include "TInterpreter.h"
#include "TSystem.h"
#include "TLeafObject.h"
#include "TH1F.h"
#include "TROOT.h"

#include "gtest/gtest.h"

#include <vector>

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
   const auto ref = "******************************************************************************\n"
                    "*Tree    :t         : t                                                      *\n"
                    "*Entries :    10000 : Total =           40973 bytes  File  Size =        202 *\n"
                    "*        :          : Tree compression factor = 118.46                       *\n"
                    "******************************************************************************\n"
                    "Cluster Range #  Entry Start      Last Entry           Size   Number of clusters\n"
                    "0                0                9999                 5966          2\n"
                    "Total number of clusters: 2 \n"; // This was 1 before the fix
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
#define TO_LITERAL(string) _QUOTE_(string)

TEST(TTreeRegressions, TTreeFormulaMemberIndex)
{
   gInterpreter->Declare(TO_LITERAL(MYSUBCLASS));
   gInterpreter->Declare(TO_LITERAL(MYCLASS));

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
