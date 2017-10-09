#include "ROOT/TDataFrame.hxx"
#include "ROOT/TTrivialDS.hxx"
#include "TMemFile.h"
#include "TTree.h"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;

TEST(TDataFrameInterface, CreateFromCStrings)
{
   TDataFrame tdf("t", "file");
}

TEST(TDataFrameInterface, CreateFromStrings)
{
   std::string t("t"), f("file");
   TDataFrame tdf(t, f);
}

TEST(TDataFrameInterface, CreateFromContainer)
{
   std::string t("t");
   std::vector<std::string> f({"f1", "f2"});
   TDataFrame tdf(t, f);
}

TEST(TDataFrameInterface, CreateFromInitList)
{
   TDataFrame tdf("t", {"f1", "f2"});
}

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

TEST(TDataFrameInterface, CreateAliases)
{
   TDataFrame tdf(1);
   auto aliased_tdf = tdf.Define("c0", []() { return 0; }).Alias("c1", "c0").Alias("c2", "c0").Alias("c3", "c1");
   auto c = aliased_tdf.Count();
   EXPECT_EQ(1U, *c);

   int ret(1);
   try {
      aliased_tdf.Alias("c4", "c");
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when trying to alias a non-existing column.";

   ret = 1;
   try {
      aliased_tdf.Alias("c0", "c2");
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when specifying an alias name which is the name of a column.";

   ret = 1;
   try {
      aliased_tdf.Alias("c2", "c1");
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when re-using an alias for a different column.";
}

TEST(TDataFrameInterface, CheckAliasesPerChain)
{
   TDataFrame tdf(1);
   auto d = tdf.Define("c0", []() { return 0; });
   // Now branch the graph
   auto ok = []() { return true; };
   auto f0 = d.Filter(ok);
   auto f1 = d.Filter(ok);
   auto f0a = f0.Alias("c1", "c0");
   // must work
   auto f0aa = f0a.Alias("c2", "c1");
   // must fail
   auto ret = 1;
   try {
      auto f1a = f1.Alias("c2", "c1");
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "No exception thrown when trying to alias a non-existing column.";
}

TEST(TDataFrameInterface, GetColumnNamesFromScratch)
{
   TDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto names = f.Define("a", dummyGen).Define("b", dummyGen).Define("tdfDummy_", dummyGen).GetColumnNames();
   EXPECT_STREQ("a", names[0].c_str());
   EXPECT_STREQ("b", names[1].c_str());
   EXPECT_EQ(2U, names.size());
}

TEST(TDataFrameInterface, GetColumnNamesFromTree)
{
   TTree t("t", "t");
   int a, b;
   t.Branch("a", &a);
   t.Branch("b", &b);
   TDataFrame tdf(t);
   auto names = tdf.GetColumnNames();
   EXPECT_STREQ("a", names[0].c_str());
   EXPECT_STREQ("b", names[1].c_str());
   EXPECT_EQ(2U, names.size());
}

TEST(TDataFrameInterface, GetColumnNamesFromSource)
{
   std::unique_ptr<TDataSource> tds(new TTrivialDS(1));
   TDataFrame tdf(std::move(tds));
   auto names = tdf.Define("b", []() { return 1; }).GetColumnNames();
   EXPECT_STREQ("b", names[0].c_str());
   EXPECT_STREQ("col0", names[1].c_str());
   EXPECT_EQ(2U, names.size());
}

TEST(TDataFrameInterface, DefaultColumns)
{
   TDataFrame tdf(8);
   ULong64_t i(0ULL);
   auto checkSlotAndEntries = [&i](unsigned int slot, ULong64_t entry) {
      EXPECT_EQ(entry, i);
      EXPECT_EQ(slot, 0U);
      i++;
   };
   tdf.Foreach(checkSlotAndEntries, {"tdfslot_", "tdfentry_"});
}

TEST(TDataFrameInterface, JitDefaultColumns)
{
   TDataFrame tdf(8);
   auto f = tdf.Filter("tdfslot_ + tdfentry_ == 3");
   auto maxEntry = f.Max("tdfentry_");
   auto minEntry = f.Min("tdfentry_");
   EXPECT_EQ(*maxEntry, *minEntry);
}
