#include "ROOT/RDataFrame.hxx"
#include "ROOT/RTrivialDS.hxx"
#include "TMemFile.h"
#include "TTree.h"

#include "gtest/gtest.h"

using namespace ROOT;
using namespace ROOT::RDF;

TEST(RDataFrameInterface, CreateFromCStrings)
{
   RDataFrame tdf("t", "file");
}

TEST(RDataFrameInterface, CreateFromStrings)
{
   std::string t("t"), f("file");
   RDataFrame tdf(t, f);
}

TEST(RDataFrameInterface, CreateFromContainer)
{
   std::string t("t");
   std::vector<std::string> f({"f1", "f2"});
   RDataFrame tdf(t, f);
}

TEST(RDataFrameInterface, CreateFromInitList)
{
   RDataFrame tdf("t", {"f1", "f2"});
}

TEST(RDataFrameInterface, CreateFromNullTDirectory)
{
   int ret = 1;
   try {
      RDataFrame tdf("t", nullptr);
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameInterface, CreateFromNonExistingTree)
{
   int ret = 1;
   try {
      RDataFrame tdf("theTreeWhichDoesNotExist", gDirectory);
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret);
}

TEST(RDataFrameInterface, CreateFromTree)
{
   TMemFile f("dataframe_interfaceAndUtils_0.root", "RECREATE");
   TTree t("t", "t");
   RDataFrame tdf(t);
   auto c = tdf.Count();
   EXPECT_EQ(0U, *c);
}

TEST(RDataFrameInterface, CreateAliases)
{
   RDataFrame tdf(1);
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

TEST(RDataFrameInterface, CheckAliasesPerChain)
{
   RDataFrame tdf(1);
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

TEST(RDataFrameInterface, GetColumnNamesFromScratch)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto names = f.Define("a", dummyGen).Define("b", dummyGen).Define("tdfDummy_", dummyGen).GetColumnNames();
   EXPECT_STREQ("a", names[0].c_str());
   EXPECT_STREQ("b", names[1].c_str());
   EXPECT_EQ(2U, names.size());
}

TEST(RDataFrameInterface, GetColumnNamesFromTree)
{
   TTree t("t", "t");
   int a, b;
   t.Branch("a", &a);
   t.Branch("b", &b);
   RDataFrame tdf(t);
   auto names = tdf.GetColumnNames();
   EXPECT_STREQ("a", names[0].c_str());
   EXPECT_STREQ("a.a", names[1].c_str());
   EXPECT_STREQ("b", names[2].c_str());
   EXPECT_STREQ("b.b", names[3].c_str());
   EXPECT_EQ(4U, names.size());
}

TEST(RDataFrameInterface, GetColumnNamesFromOrdering)
{
   TTree t("t", "t");
   int a, b;
   t.Branch("zzz", &a);
   t.Branch("aaa", &b);
   RDataFrame tdf(t);
   auto names = tdf.GetColumnNames();
   EXPECT_STREQ("zzz", names[0].c_str());
   EXPECT_STREQ("zzz.zzz", names[1].c_str());
   EXPECT_STREQ("aaa", names[2].c_str());
   EXPECT_STREQ("aaa.aaa", names[3].c_str());
   EXPECT_EQ(4U, names.size());
}

TEST(RDataFrameInterface, GetColumnNamesFromSource)
{
   std::unique_ptr<RDataSource> tds(new RTrivialDS(1));
   RDataFrame tdf(std::move(tds));
   auto names = tdf.Define("b", []() { return 1; }).GetColumnNames();
   EXPECT_STREQ("b", names[0].c_str());
   EXPECT_STREQ("col0", names[1].c_str());
   EXPECT_EQ(2U, names.size());
}

TEST(RDataFrameInterface, DefaultColumns)
{
   RDataFrame tdf(8);
   ULong64_t i(0ULL);
   auto checkSlotAndEntries = [&i](unsigned int slot, ULong64_t entry) {
      EXPECT_EQ(entry, i);
      EXPECT_EQ(slot, 0U);
      i++;
   };
   tdf.Foreach(checkSlotAndEntries, {"tdfslot_", "tdfentry_"});
}

TEST(RDataFrameInterface, JitDefaultColumns)
{
   RDataFrame tdf(8);
   auto f = tdf.Filter("tdfslot_ + tdfentry_ == 3");
   auto maxEntry = f.Max("tdfentry_");
   auto minEntry = f.Min("tdfentry_");
   EXPECT_EQ(*maxEntry, *minEntry);
}

TEST(RDataFrameInterface, InvalidDefine)
{
   RDataFrame df(1);
   try {
      df.Define("1", [] { return true; });
   } catch (const std::runtime_error &e) {
      EXPECT_STREQ("Cannot define column \"1\": not a valid C++ variable name.", e.what());
   }
   try {
      df.Define("a-b", "true");
   } catch (const std::runtime_error &e) {
      EXPECT_STREQ("Cannot define column \"a-b\": not a valid C++ variable name.", e.what());
   }
}

TEST(RDataFrameInterface, Upcasting)
{
   ROOT::RDataFrame df(10);
   std::vector<ROOT::RDF::RInterface<ROOT::Detail::RDF::RFilterBase>> v;
   v.emplace_back(df.Filter([] { return true; }));
   EXPECT_EQ(*v[0].Count(), 10u);
}
