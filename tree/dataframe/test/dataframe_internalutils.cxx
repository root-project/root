#include <stdexcept>
#include <string>
#include <vector>

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDFHelpers.hxx"
#include "ROOT/RDF/InternalUtils.hxx"
#include "TSystem.h"
#include "TTree.h"

#include "gtest/gtest.h"

void EXPECT_VEC_EQ(const std::vector<std::string> &vec1, const std::vector<std::string> &vec2)
{
   ASSERT_EQ(vec1.size(), vec2.size());
   for (std::size_t i = 0u; i < vec1.size(); ++i)
      EXPECT_EQ(vec1[i], vec2[i]);
}

TEST(RDataFrameInternalUtils, CheckTreeInfo)
{
   std::string treename{"tree_1"};
   const char *filename = "dataframe_internalutils_file_1.root";

   {
      ROOT::RDataFrame df{1};
      df.Define("x", []() -> int { return 42; }).Snapshot<int>(treename, filename, {"x"});
   }

   ROOT::RDataFrame df{treename, filename};

   auto ti = ROOT::Internal::RDF::MakeTreeInfo(ROOT::RDF::AsRNode(df));
   ASSERT_TRUE(ti != nullptr);
   EXPECT_EQ(ti->fTreeName, treename);
   EXPECT_VEC_EQ(ti->fFileNames, {std::string{filename}});
   EXPECT_VEC_EQ(ti->fTreeNamesInFiles, {treename});

   gSystem->Unlink(filename);
}

TEST(RDataFrameInternalUtils, NoTreeInfo)
{
   ROOT::RDataFrame df{1};
   auto ti = ROOT::Internal::RDF::MakeTreeInfo(ROOT::RDF::AsRNode(df));
   ASSERT_TRUE(ti == nullptr);
}
