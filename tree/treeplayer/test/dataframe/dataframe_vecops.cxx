#include <ROOT/TDataFrame.hxx>
#include <ROOT/TVec.hxx>
#include <TBranchElement.h>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h> // Unlink
#include <gtest/gtest.h>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::VecOps;

template <typename T>
auto All(const TVec<T> &v) -> decltype(bool(v[0]))
{
   for (auto &e : v)
      if (!e)
         return false;
   return true;
}

TEST(TDFAndVecOps, ReadStdVectorAsTVec)
{
   const auto fname = "tdfandvecops.root";
   const auto treename = "t";
   const auto nEntries = 5u;
   // write out a small file with an std::vector column
   auto makeStdVec = []() { return std::vector<int>({1, 2, 3}); };
   TDataFrame(nEntries).Define("v", makeStdVec).Snapshot<std::vector<int>>(treename, fname, {"v"});

   // read it from a non-jitted action
   TDataFrame d(treename, fname);
   auto checkTVec = [](const TVec<int> &v) {
      EXPECT_EQ(v.size(), 3u);
      EXPECT_TRUE(All(v == TVec<int>{1, 2, 3}));
   };
   d.Foreach(checkTVec, {"v"});

   // read it from a jitted string as a TVec
   // filter string would be invalid if v was read as a std::vector
   auto filterStr = "ROOT::Experimental::VecOps::TVec<int> v2 = Map(v, [](int i) { return i*i; }); return true;";
   auto c = d.Filter(filterStr).Count();
   EXPECT_EQ(*c, nEntries);

   gSystem->Unlink(fname);
}

TEST(TDFAndVecOps, DefineTVec)
{
   auto makeTVec = []() { return TVec<double>({1., 2., 3.}); };
   auto max = *TDataFrame(1).Define("v", makeTVec).Max<TVec<double>>("v");
   EXPECT_DOUBLE_EQ(max, 3.);
}

TEST(TDFAndVecOps, SnapshotTVec)
{
   // write TVec to file
   const auto fname = "tdfandvecops.root";
   const auto treename = "t";
   const auto nEntries = 5u;
   auto makeTVec = []() { return TVec<int>({1, 2, 3}); };
   TDataFrame(nEntries).Define("v", makeTVec).Snapshot<TVec<int>>(treename, fname, {"v"});

   // check the TVec was written as a std::vector
   TFile f(fname);
   auto t = static_cast<TTree *>(f.Get(treename));
   auto b = static_cast<TBranchElement *>(t->GetBranch("v"));
   ASSERT_TRUE(b != nullptr);
   auto branchTypeName = b->GetClassName();
   EXPECT_EQ(branchTypeName, "vector<int>");

   gSystem->Unlink(fname);
}
