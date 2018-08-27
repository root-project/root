#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RVec.hxx>

#include <algorithm>
#include <deque>
#include <vector>
#include <sys/stat.h>
#include <fstream>

#include "gtest/gtest.h"
using namespace ROOT;
using namespace ROOT::RDF;
using namespace ROOT::VecOps;

struct TrueFunctor {
   bool operator()() const { return true; }
};

bool trueFunction()
{
   return true;
}

TEST(RDFHelpers, Not)
{
   // Not(lambda)
   auto l = []() { return true; };
   EXPECT_EQ(Not(l)(), !l());
   // Not(functor)
   TrueFunctor t;
   auto falseFunctor = Not(t);
   EXPECT_EQ(falseFunctor(), false);
   EXPECT_EQ(Not(TrueFunctor())(), false);
   // Not(freeFunction)
   EXPECT_EQ(Not(trueFunction)(), false);

   // Not+RDF
   EXPECT_EQ(1u, *RDataFrame(1).Filter(Not(Not(l))).Count());
}

TEST(RDFHelpers, PassAsVec)
{
   auto One = [] { return 1; };
   auto df = RDataFrame(1).Define("one", One).Define("_1", One);

   auto TwoOnes = [](const std::vector<int> &v) { return v.size() == 2 && v[0] == 1 && v[1] == 1; };
   EXPECT_EQ(1u, *df.Filter(PassAsVec<2, int>(TwoOnes), {"one", "_1"}).Count());
   auto TwoOnesRVec = [](const RVec<int> &v) { return v.size() == 2 && All(v == 1); };
   EXPECT_EQ(1u, *df.Filter(PassAsVec<2, int>(TwoOnesRVec), {"one", "_1"}).Count());
   auto TwoOnesDeque = [](const std::deque<int> &v) { return v.size() == 2 && v[0] == 1 && v[1] == 1; };
   EXPECT_EQ(1u, *df.Filter(PassAsVec<2, int>(TwoOnesDeque), {"one", "_1"}).Count());
}

TEST(RDFHelpers, ListFilesInDirectory)
{
   // Create directory
   mkdir("test_RDFHelpers_CreateFilelist", S_IRWXU);

   // Create some files
   std::ofstream file1("test_RDFHelpers_CreateFilelist/a.log");
   file1.close();
   std::ofstream file2("test_RDFHelpers_CreateFilelist/b.log");
   file2.close();
   std::ofstream file3("test_RDFHelpers_CreateFilelist/c.txt");
   file3.close();

   // Create filelist with all files
   auto list1 = CreateFilelist("test_RDFHelpers_CreateFilelist/");
   EXPECT_EQ(3u, list1.size());
   EXPECT_EQ("test_RDFHelpers_CreateFilelist/a.log", list1[0]);
   EXPECT_EQ("test_RDFHelpers_CreateFilelist/b.log", list1[1]);
   EXPECT_EQ("test_RDFHelpers_CreateFilelist/c.txt", list1[2]);

   // Create filelist with given hint
   auto list2 = CreateFilelist("test_RDFHelpers_CreateFilelist/", ".log");
   EXPECT_EQ(2u, list2.size());
   EXPECT_EQ("test_RDFHelpers_CreateFilelist/a.log", list2[0]);
   EXPECT_EQ("test_RDFHelpers_CreateFilelist/b.log", list2[1]);

   // Test joining input path to filename without trailing slash
   auto list3 = CreateFilelist("test_RDFHelpers_CreateFilelist", ".txt");
   EXPECT_EQ(1u, list3.size());
   EXPECT_EQ("test_RDFHelpers_CreateFilelist/c.txt", list3[0]);
}
