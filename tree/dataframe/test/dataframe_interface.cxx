#include "CounterHelper.h"
#include "MaxSlotHelper.h"

#include "ROOT/RCsvDS.hxx"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RStringView.hxx"
#include "ROOT/RTrivialDS.hxx"
#include "TMemFile.h"
#include "TSystem.h"
#include "TTree.h"

#include "gtest/gtest.h"

#include <thread>

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
   EXPECT_ANY_THROW(aliased_tdf.Alias("a-b", "c1")) << "No exception thrown when the alias is not a valid C++ variable name.";
}

TEST(RDataFrameInterface, CheckAliasesPerChain)
{
   RDataFrame tdf(1);
   auto d = tdf.Define("c0", []() { return 42; });
   // Now branch the graph
   auto ok = []() { return true; };
   auto f0 = d.Filter(ok);
   auto f1 = d.Filter(ok);
   auto f0a = f0.Alias("c1", "c0");
   // must work
   auto f0aa = f0a.Alias("c2", "c1");
   EXPECT_EQ(f0aa.Max<int>("c2").GetValue(), 42);
   // must fail
   EXPECT_ANY_THROW(f1.Alias("c2", "c1")) << "No exception thrown when trying to alias a non-existing column.";
}

TEST(RDataFrameInterface, PerBranchAliases)
{
   // test that it's possible to register the same alias in different branches of the computation graph
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 42; }).Define("y", [] { return 0; });
   auto dfzx = df.Alias("z", "x");
   auto dfzy = df.Alias("z", "y");

   EXPECT_ANY_THROW(df.Max<int>("z"))
      << "No exception thrown when trying to access an alias that is not present at this point of the graph.";

   auto max42 = dfzx.Max<int>("z");
   auto max0 = dfzy.Max<int>("z");
   EXPECT_EQ(*max42, 42);
   EXPECT_EQ(*max0, 0);
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
   EXPECT_STREQ("aaa", names[0].c_str());
   EXPECT_STREQ("zzz", names[1].c_str());
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
      EXPECT_STREQ("RDataFrame::Define: cannot define column \"1\". Not a valid C++ variable name.", e.what());
   }
   try {
      df.Define("a-b", "true");
   } catch (const std::runtime_error &e) {
      EXPECT_STREQ("RDataFrame::Define: cannot define column \"a-b\". Not a valid C++ variable name.", e.what());
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
   unsigned int nslots = std::min(3U, std::thread::hardware_concurrency());
   ROOT::EnableImplicitMT(nslots);
   ROOT::RDataFrame df3(1);
   EXPECT_EQ(nslots, df3.GetNSlots());
   ROOT::DisableImplicitMT();
   ROOT::RDataFrame df1(1);
   EXPECT_EQ(1U, df1.GetNSlots());
#endif
}

// ROOT-10043
TEST(RDataFrameInterface, DefineAliasedColumn)
{
   ROOT::RDataFrame rdf(1);
   auto r0 = rdf.Define("myVar", [] { return 1; });
   auto r1 = r0.Alias("newVar", "myVar");
   auto mdefine = r0.Define("newVar", [] { return 42; }).Max<int>("newVar");
   auto malias = r1.Max<int>("newVar");
   EXPECT_EQ(*mdefine, 42);
   EXPECT_EQ(*malias, 1);
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
#if defined(_WIN64)
const std::string symbol = "`private: virtual void __cdecl RDataFrameInterface_TypeUnknownToInterpreter_Test::TestBody(void) __ptr64'::`2'::SimpleType";
#elif defined(_WIN32)
const std::string symbol = "`private: virtual void __thiscall RDataFrameInterface_TypeUnknownToInterpreter_Test::TestBody(void)'::`2'::SimpleType";
#else
const std::string symbol = "RDataFrameInterface_TypeUnknownToInterpreter_Test::TestBody()::SimpleType";
#endif

TEST(RDataFrameInterface, TypeUnknownToInterpreter)
{
   struct SimpleType {
      double a;
      double b;
   };

   auto make_s = [] { return SimpleType{0, 0}; };
   auto df = ROOT::RDataFrame(1).Define("res", make_s);
   bool hasThrown = false;
   std::stringstream ss;
   ss << "The type of custom column \"res\" (" << symbol << ") is not known to the interpreter, " <<
         "but a just-in-time-compiled Snapshot call requires this column. Make sure to create " <<
         "and load ROOT dictionaries for this column's class.";
   EXPECT_RUNTIME_ERROR_WITH_MSG(
      df.Snapshot("result", "RESULT2.root"),
      ss.str().c_str());
   ss.str("");
   ss << "The type of custom column \"res\" (" << symbol << ") is not known to the interpreter, " <<
      "but a just-in-time-compiled Define call requires this column. Make sure to create and " <<
      "load ROOT dictionaries for this column's class.";
   EXPECT_RUNTIME_ERROR_WITH_MSG(
      df.Define("res2", "res"),
      ss.str().c_str());
   ss.str("");
   ss << "The type of custom column \"res\" (" << symbol << ") is not known to the interpreter, " <<
      "but a just-in-time-compiled Filter call requires this column. Make sure to create and " <<
      "load ROOT dictionaries for this column's class.";
   EXPECT_RUNTIME_ERROR_WITH_MSG(
      df.Filter("res; return true;"),
      ss.str().c_str());
}

// ROOT-10942
TEST(RDataFrameInterface, ColumnWithSimpleStruct)
{
   gInterpreter->Declare("struct S { int a; int b; };");
   S c;
   c.a = 42;
   c.b = 2;
   TTree t("t", "t");
   t.Branch("c", &c);
   t.Fill();

   EXPECT_EQ(t.GetLeaf("a"),t.GetLeaf("c.a"));
   EXPECT_EQ(t.GetLeaf("b"),t.GetLeaf("c.b"));
   EXPECT_NE(t.GetLeaf("c.a"),t.GetLeaf("c.b"));
   EXPECT_NE(t.GetLeaf("c.b"),nullptr);

   ROOT::RDataFrame df(t);
   const std::vector<std::string> expected({ "a",  "b", "c", "c.a", "c.b" });
   EXPECT_EQ(df.GetColumnNames(), expected);
   for (std::string_view col : {"c.a", "a"}) {
      EXPECT_DOUBLE_EQ(df.Mean<int>(col).GetValue(), 42.); // compiled
      EXPECT_DOUBLE_EQ(df.Mean(col).GetValue(), 42.); // jitted
   }
}

// Issue #6435
TEST(RDataFrameInterface, MinMaxSumMeanStdDevOfScalar)
{
   auto df = ROOT::RDataFrame(4).Range(1, 0).Define("x", [](ULong64_t e) { return int(e); }, {"rdfentry_"});
   auto max = df.Max<int>("x");
   auto jit_max = df.Max("x");
   auto min = df.Min<int>("x");
   auto jit_min = df.Min("x");
   auto sum = df.Sum<int>("x");
   auto jit_sum = df.Sum("x");
   auto mean = df.Mean<int>("x");
   auto jit_mean = df.Mean("x");
   auto stddev = df.StdDev<int>("x");
   auto jit_stddev = df.StdDev("x");

   EXPECT_EQ(*max, 3);
   EXPECT_DOUBLE_EQ(*jit_max, 3.f);
   EXPECT_EQ(*min, 1);
   EXPECT_DOUBLE_EQ(*jit_min, 1.f);
   EXPECT_EQ(*sum, 6);
   EXPECT_DOUBLE_EQ(*jit_sum, 6.f);
   EXPECT_DOUBLE_EQ(*mean, 2);
   EXPECT_DOUBLE_EQ(*jit_mean, 2);
   EXPECT_DOUBLE_EQ(*stddev, 1.f);
   EXPECT_DOUBLE_EQ(*jit_stddev, 1.f);
}

TEST(RDataFrameInterface, MinMaxSumMeanStdDevOfRVec)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return ROOT::RVec<int>{1,2,3}; });
   auto max = df.Max<ROOT::RVec<int>>("x");
   auto jit_max = df.Max("x");
   auto min = df.Min<ROOT::RVec<int>>("x");
   auto jit_min = df.Min("x");
   auto sum = df.Sum<ROOT::RVec<int>>("x");
   auto jit_sum = df.Sum("x");
   auto mean = df.Mean<ROOT::RVec<int>>("x");
   auto jit_mean = df.Mean("x");
   auto stddev = df.StdDev<ROOT::RVec<int>>("x");
   auto jit_stddev = df.StdDev("x");

   EXPECT_EQ(*max, 3);
   EXPECT_DOUBLE_EQ(*jit_max, 3.f);
   EXPECT_EQ(*min, 1);
   EXPECT_DOUBLE_EQ(*jit_min, 1.f);
   EXPECT_EQ(*sum, 6);
   EXPECT_DOUBLE_EQ(*jit_sum, 6.f);
   EXPECT_DOUBLE_EQ(*mean, 2);
   EXPECT_DOUBLE_EQ(*jit_mean, 2);
   EXPECT_DOUBLE_EQ(*stddev, 1.f);
   EXPECT_DOUBLE_EQ(*jit_stddev, 1.f);
}

class Product {
public:
   Product() : _x(0), _y(0) {}
   Product(double x, double y) : _x(x), _y(y) {}
   ~Product() {}

   double GetProduct() { return _x * _y; }

private:
   double _x, _y;
};

// ROOT-10273, using jitting when some non-jitted types are unknown to the intepreter
TEST(RDataFrameInterface, JittingAndNonJittedTypes)
{
   auto df = ROOT::RDataFrame(10)
                .Define("x", "1.")
                .Define("y", "2.")
                .Define("products", [](double x, double y) { return Product(x, y); }, {"x", "y"})
                .Define("moreproducts", [](double x, double y) { return std::vector<Product>(10, Product(x, y)); },
                        {"x", "y"});

   df.Foreach([](Product &p) { EXPECT_EQ(p.GetProduct(), 2); }, {"products"});
}


// GitHub issue #7058
TEST(RDataFrameInterface, GetColumnTypeOfAlias)
{
   // from a Define
   {
      auto df = ROOT::RDataFrame(1).Define("x", [] { return 0; }).Alias("y", "x");
      EXPECT_EQ(df.GetColumnType("y"), "int");
   }

   // from a branch
   {
      TTree t("t", "t");
      int x = 42;
      t.Branch("x", &x);
      t.Fill();
      auto df = ROOT::RDataFrame(t).Alias("y", "x");
      EXPECT_EQ(df.GetColumnType("y"), "Int_t");
   }
}

TEST(RDataFrameInterface, JittedExprWithMultipleReturns)
{
   const auto counts = ROOT::RDataFrame(1)
                          .Define("x", [] { return 42; })
                          .Filter("if (x == 42) { return true; } else { return false; }")
                          .Count()
                          .GetValue();
   EXPECT_EQ(counts, 1ull);
}

TEST(RDataFrameInterface, JittedExprWithManyVars)
{
   std::string expr = "x + x + x + x";
   for (int i = 0; i < 10; ++i) {
      expr = expr + '+' + expr;
   }
   expr = expr + ">0";
   const auto counts = ROOT::RDataFrame(1)
                          .Define("x", [] { return 1; })
                          .Filter(expr)
                          .Count()
                          .GetValue();
   EXPECT_EQ(counts, 1ull);
}

TEST(RDataFrameInterface, Describe)
{
   // empty dataframe
   RDataFrame df1(1);
   const auto ref1 = "Empty dataframe filling 1 row\n"
                     "\n"
                     "Property                Value\n"
                     "--------                -----\n"
                     "Columns in total            0\n"
                     "Columns from defines        0\n"
                     "Event loops run             0\n"
                     "Processing slots            1\n"
                     "\n"
                     "Column  Type    Origin\n"
                     "------  ----    ------\n";
   EXPECT_EQ(df1.Describe().AsString(), ref1);

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   std::cout << df1.Describe();
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);
   EXPECT_EQ(strCout.str(), ref1);

   // create in-memory tree
   TTree tree("tree", "tree");
   int myInt = 1u;
   float myFloat = 1.f;
   tree.Branch("myInt", &myInt, "myInt/I");
   tree.Branch("myFloat", &myFloat, "myFloat/F");
   tree.Fill();

   // dataframe with various data types
   RDataFrame df2(tree);
   auto df3 = df2.Define("myVec", "ROOT::RVec<float>({1, 2, 3})")
                 .Define("myLongColumnName", "1u");
   df3.Sum("myInt").GetValue(); // trigger the event loop once
   const auto ref2 = "Dataframe from TTree tree (in-memory)\n"
                     "\n"
                     "Property                Value\n"
                     "--------                -----\n"
                     "Columns in total            4\n"
                     "Columns from defines        2\n"
                     "Event loops run             1\n"
                     "Processing slots            1\n"
                     "\n"
                     "Column                  Type                            Origin\n"
                     "------                  ----                            ------\n"
                     "myFloat                 Float_t                         Dataset\n"
                     "myInt                   Int_t                           Dataset\n"
                     "myLongColumnName        unsigned int                    Define\n"
                     "myVec                   ROOT::VecOps::RVec<float>       Define";
   EXPECT_EQ(df3.Describe().AsString(), ref2);
}

TEST(RDFSimpleTests, LeafWithDifferentNameThanBranch)
{
   TTree t("t", "t");
   int x = 42;
   t.Branch("x", &x, "y/I");
   t.Fill();

   auto m = ROOT::RDataFrame(t).Max<int>("x");
   EXPECT_EQ(*m, 42);
}

TEST(RDataFrameInterface, DescribeShortFormat)
{
   // trivial/empty datasource
   ROOT::RDataFrame df1a(1);
   EXPECT_EQ(df1a.Describe().AsString(/*shortFormat =*/true), "Empty dataframe filling 1 row");

   // Testing the std output printing
   std::cout << std::flush;
   // Redirect cout.
   std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();
   std::ostringstream strCout;
   std::cout.rdbuf(strCout.rdbuf());
   df1a.Describe().Print(/*shortFormat =*/true);
   // Restore old cout.
   std::cout.rdbuf(oldCoutStreamBuf);
   EXPECT_EQ(strCout.str(), "Empty dataframe filling 1 row");

   ROOT::RDataFrame df1b(2);
   EXPECT_EQ(df1b.Describe().AsString(/*shortFormat =*/true), "Empty dataframe filling 2 rows");

   // ttree/tchain
   // case: in-memory tree
   TTree tree("someName", "someTitle");
   ROOT::RDataFrame df2a(tree);
   EXPECT_EQ(df2a.Describe().AsString(/*shortFormat =*/true), "Dataframe from TTree someName (in-memory)");

   {
      // case: ctor from a single file
      TFile f("testDescribeDataset1.root", "recreate");
      TTree t("myTree", "foo");
      t.Write();
   }
   ROOT::RDataFrame df2b("myTree", "testDescribeDataset1.root");
   // NOTE: using the RDataFrame("tree", "file.root") ctor, it's always a TChain
   std::string ss1 = "Dataframe from TChain myTree in file testDescribeDataset1.root";
   EXPECT_EQ(df2b.Describe().AsString(/*shortFormat =*/true), ss1);

   // case: ctor with multiple files
   {
      TFile f("testDescribeDataset2.root", "recreate");
      TTree t("myTree", "foo");
      t.Write();
   }
   ROOT::RDataFrame df2d("myTree", {"testDescribeDataset1.root", "testDescribeDataset2.root"});
   std::string ss2 = "Dataframe from TChain myTree in files\n  testDescribeDataset1.root\n  testDescribeDataset2.root";
   EXPECT_EQ(df2d.Describe().AsString(/*shortFormat =*/true), ss2);

   // case: ttree/tchain with friends
   {
      TFile f("testDescribeDataset3.root", "recreate");
      TTree t("myTree", "foo");
      t.Write();
   }
   TFile f4("testDescribeDataset1.root");
   auto t4 = f4.Get<TTree>("myTree");
   TFile f5("testDescribeDataset2.root");
   auto t5 = f5.Get<TTree>("myTree");
   TFile f6("testDescribeDataset3.root");
   auto t6 = f6.Get<TTree>("myTree");
   TChain chain1("myTree");
   chain1.AddFile("testDescribeDataset2.root");
   chain1.AddFile("testDescribeDataset3.root");
   t4->AddFriend(t5);
   t4->AddFriend(t6, "myAlias");
   t4->AddFriend(&chain1, "myAlias2");
   ROOT::RDataFrame df2e(*t4);
   auto ss3 = std::string("Dataframe from TTree myTree in file testDescribeDataset1.root\nwith friends\n") +
              "  myTree testDescribeDataset2.root\n  myTree (myAlias) testDescribeDataset3.root\n" +
              "  myTree (myAlias2)\n    myTree testDescribeDataset2.root\n    myTree testDescribeDataset3.root";
   EXPECT_EQ(df2e.Describe().AsString(/*shortFormat =*/true), ss3);

   // others with an actual fDataSource, like csv
   auto df3 = ROOT::RDF::MakeCsvDataFrame("RCsvDS_test_headers.csv");
   EXPECT_EQ(df3.Describe().AsString(/*shortFormat =*/true), "Dataframe from datasource RCsv");

   for (int i = 1; i <= 3; ++i)
      gSystem->Unlink(("testDescribeDataset" + std::to_string(i) + ".root").c_str());
}

// #var is a convenience alias for R_rdf_sizeof_var.
TEST(RDataFrameInterface, ShortSyntaxForCollectionSizes)
{
   auto df = ROOT::RDataFrame(1).Define("R_rdf_sizeof_x", [] { return 42; });
   auto m1 = df.Max<int>("#x");
   auto m2 = df.Max("#x");
   auto m3 = df.Define("y", [] (int xs) { return xs; }, {"#x"}).Max<int>("y");
   auto m4 = df.Filter("2 + pow(#x, 2) > 0").Max<int>("#x");
   auto dfWithAlias = df.Alias("szx", "#x");
   auto m5 = dfWithAlias.Max<int>("szx");
   auto m6 = dfWithAlias.Max("szx");
   EXPECT_EQ(*m1, 42);
   EXPECT_EQ(*m2, 42);
   EXPECT_EQ(*m3, 42);
   EXPECT_EQ(*m4, 42);
   EXPECT_EQ(*m5, 42);
   EXPECT_EQ(*m6, 42);
}

// make sure #pragma is ignored, and multiple #var1 #var2 are allowed
TEST(RDataFrameInterface, StressShortSyntaxForCollectionSizes)
{
   gInterpreter->Declare("#define RDF_DO_FILTER 1");
   auto df = ROOT::RDF::RNode(ROOT::RDataFrame(42));
   // Define R_rdf_sizeof_var{1,2,...,100}
   for (int i = 1; i <= 100; ++i)
      df = df.Define("R_rdf_sizeof_var" + std::to_string(i), [] { return 1; });

   // Filter expression is "#var1 + #var2 + ... + #var100 == 100"
   std::string expr = "#var1";
   for (int i = 2; i <= 100; ++i)
      expr += "+#var" + std::to_string(i);
   expr = expr + " == 100";
   expr = "\n#ifdef RDF_DO_FILTER\nreturn " + expr + ";\n#else\nreturn false;\n#endif";
   df = df.Filter(expr);
   auto c = df.Count().GetValue();
   EXPECT_EQ(c, 42ull);
}

TEST(RDataFrameInterface, MutableForeach)
{
   int i = 0;
   ROOT::RDataFrame(10).Foreach([&](ULong64_t) mutable { ++i; }, {"rdfentry_"});
   EXPECT_EQ(i, 10);
}

TEST(RDataFrameInterface, BookWithoutColumns)
{
   CounterHelper ch; // defined as a variable to exercise passing lvalues into Book
   EXPECT_EQ(ROOT::RDataFrame(3).Book<>(ch).GetValue(), 3);
   EXPECT_THROW(ROOT::RDataFrame(3).Book(MaxSlotHelper(1u)), std::logic_error);
}

TEST(RDataFrameInterface, SnapshotWithDuplicateColumns)
{
   EXPECT_THROW(
      (ROOT::RDataFrame(1).Snapshot<ULong64_t, ULong64_t>("t", "neverwritten.root", {"rdfentry_", "rdfentry_"})),
      std::logic_error);
   EXPECT_THROW((ROOT::RDataFrame(1).Snapshot("t", "neverwritten.root", {"rdfentry_", "rdfentry_"})), std::logic_error);
}

// #11002
TEST(RDataFrameUtils, RegexWithFriendsInJittedFilters)
{
   TTree t("t", "t");
   int x = 42;
   t.Branch("x", &x);
   t.Fill();
   TTree fr("fr", "fr");
   x = -42;
   fr.Branch("x", &x);
   fr.Fill();
   t.AddFriend(&fr);
   ROOT::RDataFrame df(t);
   // ensure that order of operations does not matter
   EXPECT_EQ(df.Filter("fr.x < 0 && x > 0").Count().GetValue(), 1);
   EXPECT_EQ(df.Filter("x > 0 && fr.x < 0").Count().GetValue(), 1);
}
