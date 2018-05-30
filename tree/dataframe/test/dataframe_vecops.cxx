#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TBranchElement.h>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h> // Unlink
#include <gtest/gtest.h>

using namespace ROOT::VecOps;
using namespace ROOT;

TEST(RDFAndVecOps, ReadStdVectorAsRVec)
{
   const auto fname = "rdfandvecops.root";
   const auto treename = "t";
   const auto nEntries = 5u;
   // write out a small file with an std::vector column
   auto makeStdVec = []() { return std::vector<int>({1, 2, 3}); };
   RDataFrame(nEntries).Define("v", makeStdVec).Snapshot<std::vector<int>>(treename, fname, {"v"});

   // read it from a non-jitted action
   RDataFrame d(treename, fname);
   auto checkRVec = [](const RVec<int> &v) {
      EXPECT_EQ(v.size(), 3u);
      EXPECT_TRUE(All(v == RVec<int>{1, 2, 3}));
   };
   d.Foreach(checkRVec, {"v"});

   // read it from a jitted string as a RVec
   // filter string would be invalid if v was read as a std::vector
   auto filterStr = "ROOT::VecOps::RVec<int> v2 = Map(v, [](int i) { return i*i; }); return true;";
   auto c = d.Filter(filterStr).Count();
   EXPECT_EQ(*c, nEntries);

   gSystem->Unlink(fname);
}

TEST(RDFAndVecOps, DefineRVec)
{
   auto makeRVec = []() { return RVec<double>({1., 2., 3.}); };
   auto max = *RDataFrame(1).Define("v", makeRVec).Max<RVec<double>>("v");
   EXPECT_DOUBLE_EQ(max, 3.);
}

TEST(RDFAndVecOps, SnapshotRVec)
{
   // write RVec to file
   const auto fname = "tdfandvecops.root";
   const auto treename = "t";
   const auto nEntries = 5u;
   auto makeRVec = []() { return RVec<int>({1, 2, 3}); };
   RDataFrame(nEntries).Define("v", makeRVec).Snapshot<RVec<int>>(treename, fname, {"v"});

   // check the RVec was written as a RVec
   TFile f(fname);
   auto t = static_cast<TTree *>(f.Get(treename));
   auto b = static_cast<TBranchElement *>(t->GetBranch("v"));
   ASSERT_TRUE(b != nullptr);
   auto branchTypeName = b->GetClassName();
   EXPECT_STREQ(branchTypeName, "vector<int,ROOT::Detail::VecOps::RAdoptAllocator<int> >");

   gSystem->Unlink(fname);
}
