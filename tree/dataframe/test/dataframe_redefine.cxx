#include <ROOT/RDataFrame.hxx>
#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RTrivialDS.hxx>
#include <TTree.h>

#include <gtest/gtest.h>

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

TEST(Redefine, Slot) {}
TEST(Redefine, SlotEntry) {}
