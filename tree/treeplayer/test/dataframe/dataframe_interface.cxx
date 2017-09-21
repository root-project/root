#include "ROOT/TDataFrame.hxx"
#include "TMemFile.h"
#include "TTree.h"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

TEST(TDataFrameInterface, CreateFromNullTDirectory)
{
   int ret = 1;
   try {
      TDataFrame tdf("t", nullptr);
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(TDataFrameInterface, CreateFromNonExistingTree)
{
   int ret = 1;
   try {
      TDataFrame tdf("theTreeWhichDoesNotExist", gDirectory);
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(TDataFrameInterface, CreateFromTree)
{
   TMemFile f("dataframe_interfaceAndUtils_0.root", "RECREATE");
   TTree t("t", "t");
   TDataFrame tdf(t);
   auto c = tdf.Count();
   EXPECT_EQ(0U, *c);
}
