#include "ROOT/RDataFrame.hxx"
#include "ROOT/RTrivialDS.hxx"
#include "TMemFile.h"
#include "TSystem.h"
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
   EXPECT_ANY_THROW(RDataFrame("t", nullptr));
}

TEST(RDataFrameInterface, CreateFromNonExistingTree)
{
   EXPECT_ANY_THROW(RDataFrame("theTreeWhichDoesNotExist", gDirectory));
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

   EXPECT_ANY_THROW(aliased_tdf.Alias("c4", "c")) << "No exception thrown when trying to alias a non-existing column.";
   EXPECT_ANY_THROW(aliased_tdf.Alias("c0", "c2")) << "No exception thrown when specifying an alias name which is the name of a column.";
   EXPECT_ANY_THROW(aliased_tdf.Alias("c2", "c1")) << "No exception thrown when re-using an alias for a different column.";
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
   EXPECT_ANY_THROW(f1.Alias("c2", "c1")) << "No exception thrown when trying to alias a non-existing column.";
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
   EXPECT_EQ(2U, names.size());
   EXPECT_STREQ("a", names[0].c_str());
   EXPECT_STREQ("b", names[1].c_str());
}

TEST(RDataFrameInterface, GetColumnNamesFromOrdering)
{
   TTree t("t", "t");
   int a, b;
   t.Branch("zzz", &a);
   t.Branch("aaa", &b);
   RDataFrame tdf(t);
   auto names = tdf.GetColumnNames();
   EXPECT_EQ(2U, names.size());
   EXPECT_STREQ("zzz", names[0].c_str());
   EXPECT_STREQ("aaa", names[1].c_str());

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

TEST(RDataFrameInterface, GetFilterNamesFromNode)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto dummyFilter = [](int val) { return val > 0; };
   auto names = f.Define("a", dummyGen)
                   .Define("b", dummyGen)
                   .Filter("a>0")
                   .Range(30)
                   .Filter(dummyFilter, {"a"})
                   .Define("d", dummyGen)
                   .Range(30)
                   .Filter("a>0", "filt_a_jit")
                   .Filter(dummyFilter, {"a"}, "filt_a")
                   .Filter("a>0")
                   .Filter(dummyFilter, {"a"})
                   .GetFilterNames();

   std::vector<std::string> comparison(
      {"Unnamed Filter", "Unnamed Filter", "filt_a_jit", "filt_a", "Unnamed Filter", "Unnamed Filter"});
   EXPECT_EQ(comparison, names);
}

TEST(RDataFrameInterface, GetFilterNamesFromLoopManager)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto dummyFilter = [](int val) { return val > 0; };
   auto names_one = f.Define("a", dummyGen)
                       .Define("b", dummyGen)
                       .Filter("a>0")
                       .Range(30)
                       .Filter(dummyFilter, {"a"})
                       .Define("c", dummyGen)
                       .Range(30)
                       .Filter("a>0", "filt_a_jit")
                       .Filter(dummyFilter, {"b"}, "filt_b")
                       .Filter("a>0")
                       .Filter(dummyFilter, {"a"});
   auto names_two = f.Define("d", dummyGen)
                       .Define("e", dummyGen)
                       .Filter("d>0")
                       .Range(30)
                       .Filter(dummyFilter, {"d"})
                       .Define("f", dummyGen)
                       .Range(30)
                       .Filter("d>0", "filt_d_jit")
                       .Filter(dummyFilter, {"e"}, "filt_e")
                       .Filter("e>0")
                       .Filter(dummyFilter, {"e"});

   std::vector<std::string> comparison({"Unnamed Filter", "Unnamed Filter", "filt_a_jit", "filt_b", "Unnamed Filter",
                                        "Unnamed Filter", "Unnamed Filter", "Unnamed Filter", "filt_d_jit", "filt_e",
                                        "Unnamed Filter", "Unnamed Filter"});
   auto names = f.GetFilterNames();
   EXPECT_EQ(comparison, names);
}

TEST(RDataFrameInterface, GetFilterNamesFromNodeNoFilters)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto names =
      f.Define("a", dummyGen).Define("b", dummyGen).Range(30).Define("d", dummyGen).Range(30).GetFilterNames();

   std::vector<std::string> comparison({});
   EXPECT_EQ(comparison, names);
}

TEST(RDataFrameInterface, GetFilterNamesFromLoopManagerNoFilters)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto names_one = f.Define("a", dummyGen).Define("b", dummyGen).Range(30).Define("c", dummyGen).Range(30);
   auto names_two = f.Define("d", dummyGen).Define("e", dummyGen).Range(30).Define("f", dummyGen).Range(30);

   std::vector<std::string> comparison({});
   auto names = f.GetFilterNames();
   EXPECT_EQ(comparison, names);
}

TEST(RDataFrameInterface, GetDefinedColumnNamesFromScratch)
{
   RDataFrame f(1);
   auto dummyGen = []() { return 1; };
   auto names = f.Define("a", dummyGen).Define("b", dummyGen).Define("tdfDummy_", dummyGen).GetDefinedColumnNames();
   std::sort(names.begin(), names.end());
   EXPECT_STREQ("a", names[0].c_str());
   EXPECT_STREQ("b", names[1].c_str());
   EXPECT_EQ(2U, names.size());
}

TEST(RDataFrameInterface, GetDefinedColumnNamesFromTree)
{
   TTree t("t", "t");
   int a, b;
   t.Branch("a", &a);
   t.Branch("b", &b);
   RDataFrame tdf(t);

   auto dummyGen = []() { return 1; };
   auto names = tdf.Define("d_a", dummyGen).Define("d_b", dummyGen).GetDefinedColumnNames();

   EXPECT_EQ(2U, names.size());
   std::sort(names.begin(), names.end());
   EXPECT_STREQ("d_a", names[0].c_str());
   EXPECT_STREQ("d_b", names[1].c_str());
}

TEST(RDataFrameInterface, GetDefinedColumnNamesFromSource)
{
   std::unique_ptr<RDataSource> tds(new RTrivialDS(1));
   RDataFrame tdf(std::move(tds));
   auto names = tdf.Define("b", []() { return 1; }).GetDefinedColumnNames();
   EXPECT_EQ(1U, names.size());
   EXPECT_STREQ("b", names[0].c_str());
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
   i = 0ULL;
   tdf.Foreach(checkSlotAndEntries, {"rdfslot_", "rdfentry_"});
}

TEST(RDataFrameInterface, JitDefaultColumns)
{
   RDataFrame tdf(8);
   {
      auto f = tdf.Filter("tdfslot_ + tdfentry_ == 3");
      auto maxEntry = f.Max("tdfentry_");
      auto minEntry = f.Min("tdfentry_");
      EXPECT_EQ(*maxEntry, *minEntry);
   }
   {
      auto f = tdf.Filter("rdfslot_ + rdfentry_ == 3");
      auto maxEntry = f.Max("rdfentry_");
      auto minEntry = f.Min("rdfentry_");
      EXPECT_EQ(*maxEntry, *minEntry);
   }
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

struct S {
   int a;
   int b;
};

TEST(RDataFrameInterface, GetColumnType)
{
   const auto fname = "tdf_getcolumntype.root";
   TFile f(fname, "recreate");
   TTree t("t", "t");
   S s{1,2};
   int x = 42;
   t.Branch("s", &s, "a/I:b/I");
   t.Branch("x", &x);
   t.Fill();
   t.Write();
   f.Close();

   auto df = RDataFrame("t", fname).Define("y", [] { return std::vector<int>{}; }).Define("z", "double(x)");
   EXPECT_EQ(df.GetColumnType("x"), "Int_t");
   EXPECT_EQ(df.GetColumnType("y"), "vector<int>");
   EXPECT_EQ(df.GetColumnType("z"), "double");
   EXPECT_EQ(df.GetColumnType("s.a"), "Int_t");

   gSystem->Unlink(fname);
}

TEST(RDFHelpers, CastToNode)
{
   // an empty RDF
   ROOT::RDataFrame d(1);
   ROOT::RDF::RNode n(d);
   auto n2 = ROOT::RDF::RNode(n.Filter([] { return true; }));
   auto n3 = ROOT::RDF::RNode(n2.Filter("true"));
   auto n4 = ROOT::RDF::RNode(n3.Define("x", [] { return 42; }));
   auto n5 = ROOT::RDF::RNode(n4.Define("y", "x"));
   auto n6 = ROOT::RDF::RNode(n5.Range(0,0));
   auto n7 = ROOT::RDF::RNode(n.Filter([] { return true; }, {}, "myfilter"));
   auto c = n6.Count();
   EXPECT_EQ(*c, 1ull);

   // now with a datasource
   auto df = ROOT::RDF::MakeTrivialDataFrame(10);
   auto df2 = ROOT::RDF::RNode(df.Filter([] { return true; }));
   EXPECT_EQ(*df2.Count(), 10ull);
}

// ROOT-9931
TEST(RDataFrameInterface, GraphAndHistoNoColumns)
{
   EXPECT_ANY_THROW(ROOT::RDataFrame(1).Graph()) << "No exception thrown when booking a graph with no columns available.";
   EXPECT_ANY_THROW(ROOT::RDataFrame(1).Histo1D()) << "No exception thrown when booking an histo with no columns available.";
}

// ROOT-9933
TEST(RDataFrameInterface, GetNSlots)
{
   ROOT::RDataFrame df0(1);
   EXPECT_EQ(1U, df0.GetNSlots());
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT(3);
   ROOT::RDataFrame df3(1);
   EXPECT_EQ(3U, df3.GetNSlots());
   ROOT::DisableImplicitMT();
   ROOT::RDataFrame df1(1);
   EXPECT_EQ(1U, df1.GetNSlots());
#endif
}

// ROOT-10043
TEST(RDataFrameInterface, DefineAliasedColumn)
{
   ROOT::RDataFrame rdf(1);
   auto r0 = rdf.Define("myVar", [](){return 1;});
   auto r1 = r0.Alias("newVar", "myVar");
   EXPECT_ANY_THROW(r0.Define("newVar", [](int i){return i;}, {"myVar"})) << "No exception thrown when defining a column with a name which is already an alias.";
}

// ROOT-10619
TEST(RDataFrameInterface, UnusedJittedNodes)
{
   ROOT::RDataFrame df(1);
   df.Filter("true");
   df.Define("x", "true");
   df.Foreach([]{}); // crashes if ROOT-10619 not fixed
}

#define EXPECT_RUNTIME_ERROR_WITH_MSG(expr, msg) \
   try { expr; } catch (const std::runtime_error &e) {\
      EXPECT_STREQ(e.what(), msg);\
      hasThrown = true;\
   }\
   EXPECT_TRUE(hasThrown);\
   hasThrown = false;

// ROOT-10458
TEST(RDataFrameInterface, TypeUnknownToInterpreter)
{
   struct SimpleType {
      double a;
      double b;
   };

   auto make_s = [] { return SimpleType{0, 0}; };
   auto df = ROOT::RDataFrame(1).Define("res", make_s);
   bool hasThrown = false;
   EXPECT_RUNTIME_ERROR_WITH_MSG(
      df.Snapshot("result", "RESULT2.root"),
      "The type of custom column \"res\" (RDataFrameInterface_TypeUnknownToInterpreter_Test::TestBody()::SimpleType) "
      "is not known to the interpreter, but a just-in-time-compiled Snapshot call requires this column. Make sure to "
      "create and load ROOT dictionaries for this column's class.");
   EXPECT_RUNTIME_ERROR_WITH_MSG(
      df.Define("res2", "res"),
      "The type of custom column \"res\" (RDataFrameInterface_TypeUnknownToInterpreter_Test::TestBody()::SimpleType) "
      "is not known to the interpreter, but a just-in-time-compiled Define call requires this column. Make sure to "
      "create and load ROOT dictionaries for this column's class.");
   EXPECT_RUNTIME_ERROR_WITH_MSG(
      df.Filter("res; return true;"),
      "The type of custom column \"res\" (RDataFrameInterface_TypeUnknownToInterpreter_Test::TestBody()::SimpleType) "
      "is not known to the interpreter, but a just-in-time-compiled Filter call requires this column. Make sure to "
      "create and load ROOT dictionaries for this column's class.");
}
