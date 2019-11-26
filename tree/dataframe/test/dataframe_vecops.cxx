#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <TBranchElement.h>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h> // Unlink
#include <gtest/gtest.h>
#include <vector>

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
   auto t = f.Get<TTree>(treename);
   auto b = static_cast<TBranchElement *>(t->GetBranch("v"));
   ASSERT_TRUE(b != nullptr);
   auto branchTypeName = b->GetClassName();
   EXPECT_STREQ(branchTypeName, "vector<int,ROOT::Detail::VecOps::RAdoptAllocator<int> >");

   gSystem->Unlink(fname);
}

void MakeTreeWithBools(const std::string &treename, const std::string &fname)
{
   TFile f(fname.c_str(), "recreate");
   TTree t(treename.c_str(), treename.c_str());

   int n = 0;
   bool var_arr[1000];
   bool arr[2];
   std::vector<bool> vec;

   t.Branch("vec", &vec);
   t.Branch("arr", arr, "arr[2]/O");
   t.Branch("n", &n);
   t.Branch("var_arr", var_arr, "var_arr[n]/O");

   n = 2;
   var_arr[0] = true;
   var_arr[1] = false;
   arr[0] = true;
   arr[1] = false;
   vec = {true, false};
   t.Fill();

   n = 1000;
   vec.resize(n);
   for (auto i = 0; i < n; ++i) {
      bool value = i % 2 == 0;
      var_arr[i] = value;
      vec[i] = value;
   }
   t.Fill();

   t.Write();
   f.Close();
}

TEST(RDFAndVecOps, RVecBool)
{
   const auto filename = "rdfrvecbool.root";
   const auto treename = "t";
   MakeTreeWithBools(treename, filename);

   ROOT::RDataFrame df(treename, filename);

   auto c = df.Count();
   df.Foreach(
      [](const RVec<bool> &arr, const RVec<bool> &var_arr, const RVec<bool> &vec, ULong64_t entry) {
         EXPECT_TRUE(All(arr == RVec<bool>{true, false}));
         if (entry == 0ull) {
            EXPECT_TRUE(All(var_arr == RVec<bool>{true, false}));
            EXPECT_TRUE(All(vec == RVec<bool>{true, false}));
         } else {
            const auto size = var_arr.size();
            EXPECT_EQ(size, 1000ull);
            EXPECT_EQ(size, vec.size());
            for (auto i = 0u; i < size; ++i) {
               const bool value = i % 2 == 0;
               EXPECT_EQ(var_arr[i], value);
               EXPECT_EQ(vec[i], value);
            }
         }
      },
      {"arr", "var_arr", "vec", "rdfentry_"});
   EXPECT_EQ(*c, 2ull);

   gSystem->Unlink(filename);
}

