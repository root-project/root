#include "ROOT/RDataFrame.hxx"
#include "TChain.h"
#include "TEntryList.h"
#include "TFile.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TTree.h"
#include "gtest/gtest.h"

#include <vector>

// Write to disk file filename with TTree "t" with nEntries of branch "e" assuming integer values [0..nEntries).
void MakeInputFile(const std::string &filename, int nEntries)
{
   const auto treename = "t";
   auto d = ROOT::RDataFrame(nEntries)
               .Define("e", [](ULong64_t e) { return int(e); }, {"rdfentry_"})
               .Snapshot<int>(treename, filename, {"e"});
}

class RMTRAII {
   bool fIsMT;

public:
   RMTRAII(bool isMT) : fIsMT(isMT)
   {
      if (fIsMT)
         ROOT::EnableImplicitMT();
   }
   ~RMTRAII()
   {
      if (fIsMT)
         ROOT::DisableImplicitMT();
   }
};

void TestTreeWithEntryList(bool isMT = false)
{
   const auto nEntries = 10;
   const auto treename = "t";
   const auto filename = "rdfentrylist.root";
   MakeInputFile(filename, nEntries);

   RMTRAII gomt(isMT);

   TEntryList elist("e", "e", treename, filename);
   elist.Enter(0);
   elist.Enter(nEntries - 1);

   TFile f(filename);
   auto t = f.Get<TTree>(treename);
   t->SetEntryList(&elist);

   auto entries = ROOT::RDataFrame(*t).Take<int>("e");
   EXPECT_EQ(*entries, std::vector<int>({0, nEntries - 1}));

   gSystem->Unlink(filename);
}

void TestChainWithEntryList(bool isMT = false)
{
   const auto nEntries = 10;
   const auto treename = "t";
   const auto file1 = "rdfentrylist1.root";
   const auto file2 = "rdfentrylist2.root";
   MakeInputFile(file1, nEntries);
   MakeInputFile(file2, nEntries);

   RMTRAII gomt(isMT);

   TEntryList elist1("e", "e", treename, file1);
   elist1.Enter(0);
   elist1.Enter(nEntries - 1);
   TEntryList elist2("e", "e", treename, file2);
   elist2.Enter(0);
   elist2.Enter(nEntries - 1);

   // make a TEntryList that contains two TEntryLists in its list of TEntryLists,
   // as required by TChain (see TEntryList's doc)
   TEntryList elists;
   elists.Add(&elist1);
   elists.Add(&elist2);

   TChain c(treename);
   c.Add(file1, nEntries);
   c.Add(file2, nEntries);
   c.SetEntryList(&elists);

   auto entries = ROOT::RDataFrame(c).Take<int>("e");
   EXPECT_EQ(*entries, std::vector<int>({0, nEntries - 1, 0, nEntries - 1}));

   gSystem->Unlink(file1);
   gSystem->Unlink(file2);
}

TEST(RDFEntryList, Chain)
{
   TestChainWithEntryList();
}

TEST(RDFEntryList, Tree)
{
   TestTreeWithEntryList();
}

#ifdef R__USE_IMT
TEST(RDFEntryList, ChainMT)
{
   TestChainWithEntryList(true);
}

TEST(RDFEntryList, TreeMT)
{
   TestTreeWithEntryList(true);
}
#endif
