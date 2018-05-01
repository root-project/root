#include "ROOT/TDataFrame.hxx"

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

TEST(Aliases, DefineOnAlias)
{
   TDataFrame tdf(2);
   int i = 1;
   auto m = tdf.Define("c0", [&i]() { return i++; })
               .Alias("c1", "c0")
               .Define("c2", [](int j) { return j + 1; }, {"c1"})
               .Mean<int>("c2");
   EXPECT_DOUBLE_EQ(2.5, *m);
}

TEST(Aliases, FilterOnAlias)
{
   TDataFrame tdf(2);
   int i = 1;
   auto c =
      tdf.Define("c0", [&i]() { return i++; }).Alias("c1", "c0").Filter([](int j) { return j > 1; }, {"c1"}).Count();
   EXPECT_EQ(1U, *c);
}

TEST(Aliases, DefineOnAliasJit)
{
   TDataFrame tdf(2);
   int i = 1;
   auto m = tdf.Define("c0", [&i]() { return i++; }).Alias("c1", "c0").Define("c2", "c1+1").Mean<int>("c2");
   EXPECT_DOUBLE_EQ(2.5, *m);
}

TEST(Aliases, FilterOnAliasJit)
{
   TDataFrame tdf(2);
   int i = 1;
   auto c = tdf.Define("c0", [&i]() { return i++; }).Alias("c1", "c0").Filter("c1>1").Count();
   EXPECT_EQ(1U, *c);
}
