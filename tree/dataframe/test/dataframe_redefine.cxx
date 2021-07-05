#include <ROOT/RDataFrame.hxx>
#include <ROOT/RTrivialDS.hxx>
#include <TTree.h>

#include <gtest/gtest.h>

#include <memory>

TEST(Redefine, NoJitting)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 1; });
   auto r = df.Redefine("x", [] { return 42; }).Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, Jitting)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 1; }).Define("y", "1");
   // Jitted redefine for a non-jitted Define
   auto rx = df.Redefine("x", "42").Max<int>("x");
   // Jitted redefine for a jitted Define
   auto ry = df.Redefine("y", "42").Max<int>("y");

   EXPECT_EQ(*rx, 42);
   EXPECT_EQ(*ry, 42);
}

TEST(Redefine, Branch)
{
   TTree t("t", "t");
   int x = 1;
   t.Branch("x", &x);
   t.Fill();
   auto df = ROOT::RDataFrame(t);
   auto r = df.Redefine("x", [] { return 42; }).Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, DataSourceColumn)
{
   ROOT::RDataFrame df(std::make_unique<ROOT::RDF::RTrivialDS>(1));
   auto r = df.Redefine("col0", [] { return 42; }).Max<int>("col0");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, Alias)
{
   auto df = ROOT::RDataFrame(1).Alias("x", "rdfentry_");
   EXPECT_THROW(df.Redefine("x", [] { return 42; }), std::runtime_error);
}

TEST(Redefine, Twice)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 0; }).Redefine("x", [] { return 1; }).Redefine("x", [] {
      return 42;
   });
   auto r = df.Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, Slot)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 0; }).RedefineSlot("x", [](unsigned int) { return 42; });
   auto r = df.Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, SlotEntry)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 0; }).RedefineSlotEntry("x", [](unsigned int, ULong64_t) {
      return 42;
   });
   auto r = df.Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, Parallel)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 1; });
   auto r1 = df.Redefine("x", [] { return 42; }).Max<int>("x");
   auto r2 = df.Redefine("x", [] { return 84; }).Max<int>("x");
   EXPECT_EQ(*r1, 42);
   EXPECT_EQ(*r2, 84);
}

TEST(Redefine, AliasOnRedefine)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 0; }).Redefine("x", [] { return 42; }).Alias("y", "x");
   auto r = df.Max<int>("y");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, GetColumnType)
{
   auto df = ROOT::RDataFrame(1).Define("x", [] { return 'c'; }).Redefine("x", [] { return 42; });
   EXPECT_EQ(df.GetColumnType("x"), "int");
}

TEST(Redefine, GetColumnTypeOfRedefinedBranch)
{
   TTree t("t", "t");
   int x = 1;
   t.Branch("x", &x);
   t.Fill();
   auto df = ROOT::RDataFrame(t).Redefine("x", [] { return 'c'; });
   EXPECT_EQ(df.GetColumnType("x"), "char");
}

TEST(Redefine, OriginalBranchAsInput)
{
   TTree t("t", "t");
   int x = 1;
   t.Branch("x", &x);
   t.Fill();
   auto r = ROOT::RDataFrame(t).Redefine("x", [](int _x) { return _x * 42; }, {"x"}).Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, OriginalBranchAsInputJitted)
{
   TTree t("t", "t");
   int x = 1;
   t.Branch("x", &x);
   t.Fill();
   auto r = ROOT::RDataFrame(t).Redefine("x", "x*42").Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, OriginalDefineAsInput)
{
   auto r = ROOT::RDataFrame(1)
               .Define("x", [] { return 1; })
               .Redefine("x", [](int x) { return x * 42; }, {"x"})
               .Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, OriginalDefineAsInputJitted)
{
   auto r = ROOT::RDataFrame(1)
               .Define("x", [] { return 1; })
               .Redefine("x", [](int x) { return x * 42; }, {"x"})
               .Max<int>("x");
   EXPECT_EQ(*r, 42);
}

TEST(Redefine, ErrorOnNonExistingColumn)
{
   auto df = ROOT::RDataFrame(1);
   EXPECT_THROW(df.Redefine("x", [](int x) { return x * 42; }, {"x"}), std::runtime_error);
   EXPECT_THROW(df.Redefine("x", "42"), std::runtime_error);
}
