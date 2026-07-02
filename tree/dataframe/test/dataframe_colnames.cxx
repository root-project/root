#include "ROOT/RDataFrame.hxx"
#include <TInterpreter.h>
#include <TTree.h>

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

// Test for https://github.com/root-project/root/issues/19834
TEST(ColNames, NoSource)
{
   auto df = ROOT::RDataFrame(1);
   auto df_def = df.Define("x", "int(1)");
   EXPECT_TRUE(df_def.HasColumn("x"));
   EXPECT_FALSE(df_def.HasColumn("y"));
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

// Regression test for https://github.com/root-project/root/issues/22295
// A column named "phi" must not be confused with a .phi() method call in a JIT expression.
TEST(ColNames, MethodCallNotConfusedWithColumn)
{
   // Declare minimal helper types for use in JIT strings.
   // Using gInterpreter::Declare avoids any dependency on GenVector or other libraries.
   gInterpreter->Declare("struct RDF22295Vec { double phi() const { return 42.0; } };");
   gInterpreter->Declare("struct RDF22295Mem { double phi = 1.5; };");

   // Case 1: column "phi" exists, expression calls .phi() method — must not throw,
   // and the result must be the value returned by the method (42.0), not a substitution
   // artifact. Before the fix this crashed with "no member named 'var0'" because the
   // parser replaced .phi() with .var0().
   {
      ROOT::RDataFrame df(1);
      auto df2 = df.Define("phi", "1.0");
      std::vector<double> v;
      EXPECT_NO_THROW({
         auto df3 = df2.Define("result", "RDF22295Vec{}.phi()");
         v = *df3.Take<double>("result");
      });
      EXPECT_NEAR(42.0, v[0], 1e-9);
   }

   // Case 2: column "phi" exists, expression uses BOTH .phi() method call AND standalone
   // phi column reference — only the standalone reference must be substituted.
   {
      ROOT::RDataFrame df(1);
      auto df2 = df.Define("phi", "0.5");
      std::vector<double> v;
      EXPECT_NO_THROW({
         auto df3 = df2.Define("result", "RDF22295Vec{}.phi() + phi");
         v = *df3.Take<double>("result");
      });
      // .phi() returns 42.0; standalone phi column = 0.5
      EXPECT_NEAR(42.5, v[0], 1e-9);
   }

   // Case 3: data-member access obj.phi (not a function call) must still work after the
   // fix — the preceding-dot guard must not break legitimate dot-chain expressions.
   {
      ROOT::RDataFrame df(1);
      auto df2 = df.Define("phi", "0.5");
      std::vector<double> v;
      EXPECT_NO_THROW({
         auto df3 = df2.Define("result", "RDF22295Mem{}.phi + phi");
         v = *df3.Take<double>("result");
      });
      // data member .phi = 1.5; standalone phi column = 0.5
      EXPECT_NEAR(2.0, v[0], 1e-9);
   }
}
