#include "ROOT/RDataFrame.hxx"

#include "gtest/gtest.h"


TEST(ColNames, HasColumn)
{
   // From scratch
   ROOT::RDataFrame fromScratch(1);

   EXPECT_TRUE(fromScratch.HasColumn("rdfentry_"));
   EXPECT_TRUE(fromScratch.HasColumn("rdfslot_"));

   auto rdf = fromScratch.Define("def", [](){return 0;}).Alias("alias", "def");

   EXPECT_TRUE(rdf.HasColumn("def"));
   EXPECT_TRUE(rdf.HasColumn("alias"));

   // From tree
   TTree t("t","t");
   int i;
   t.Branch("branch", &i);

   ROOT::RDataFrame fromTree(t);
   EXPECT_TRUE(fromTree.HasColumn("branch"));

   // From Source
   auto fromSource = rdf.Cache<int>({"def"});
   EXPECT_TRUE(fromSource.HasColumn("def"));

}

// ROOT-9929
TEST(ColNames, ContainedNames)
{
   TTree t("t","t");
   int i = 1;
   t.Branch("a", &i);
   t.Branch("aa", &i);
   t.Fill();

   ROOT::RDataFrame df(t);
   auto c = df.Filter("a == aa").Count();
   EXPECT_EQ(1U, *c);
}

TEST(Aliases, DefineOnAlias)
{
   ROOT::RDataFrame tdf(2);
   int i = 1;
   auto m = tdf.Define("c0", [&i]() { return i++; })
               .Alias("c1", "c0")
               .Define("c2", [](int j) { return j + 1; }, {"c1"})
               .Mean<int>("c2");
   EXPECT_DOUBLE_EQ(2.5, *m);
}

TEST(Aliases, FilterOnAlias)
{
   ROOT::RDataFrame tdf(2);
   int i = 1;
   auto c =
      tdf.Define("c0", [&i]() { return i++; }).Alias("c1", "c0").Filter([](int j) { return j > 1; }, {"c1"}).Count();
   EXPECT_EQ(1U, *c);
}

TEST(Aliases, DefineOnAliasJit)
{
   ROOT::RDataFrame tdf(2);
   int i = 1;
   auto m = tdf.Define("c0", [&i]() { return i++; }).Alias("c1", "c0").Define("c2", "c1+1").Mean<int>("c2");
   EXPECT_DOUBLE_EQ(2.5, *m);
}

TEST(Aliases, FilterOnAliasJit)
{
   ROOT::RDataFrame tdf(2);
   int i = 1;
   auto c = tdf.Define("c0", [&i]() { return i++; }).Alias("c1", "c0").Filter("c1>1").Count();
   EXPECT_EQ(1U, *c);
}
