#include "ROOT/TDataFrame.hxx"
#include "gtest/gtest.h"
using namespace ROOT::Experimental;

class TDFRanges : public ::testing::Test {
protected:
   TDFRanges() : fTDF(100) {}
   TDataFrame &GetTDF() { return fTDF; }

private:
   TDataFrame fTDF;
};

TEST_F(TDFRanges, API)
{
   auto &tdf = GetTDF();
   // all Range signatures. Event-loop is run once
   auto c1 = tdf.Range(0).Count();
   auto c2 = tdf.Range(10).Count();
   auto m = tdf.Range(5, 50).Max<ULong64_t>("tdfentry_");
   auto t = tdf.Range(5, 10, 3).Take<ULong64_t>("tdfentry_");
   EXPECT_EQ(*c1, 100u);
   EXPECT_EQ(*c2, 10u);
   EXPECT_EQ(*m, 49u);
   EXPECT_EQ(*t, std::vector<ULong64_t>({5, 8}));
}

TEST_F(TDFRanges, FromRange)
{
   auto &d = GetTDF();
   auto min = d.Range(10, 50).Range(10, 20).Min<ULong64_t>("tdfentry_");
   EXPECT_EQ(*min, 20u);
}

TEST_F(TDFRanges, FromFilter)
{
   auto &d = GetTDF();
   auto count = d.Filter([](ULong64_t b) { return b > 95; }, {"tdfentry_"}).Range(10).Count();
   EXPECT_EQ(*count, 4u);
}

TEST_F(TDFRanges, FromDefine)
{
   auto &d = GetTDF();
   auto count = d.Define("dummy", []() { return 42; }).Range(10).Count();
   EXPECT_EQ(*count, 10u);
}

TEST_F(TDFRanges, EarlyStop)
{
   auto &d = GetTDF();
   // TODO how do I check that the event-loop is actually interrupted after 20 iterations?
   unsigned int count = 0;
   auto b1 = d.Range(10).Count();

   auto b2 = d.Define("counter",
                      [&count]() {
                         ++count;
                         return 42;
                      })
                .Range(20)
                .Take<int>("counter");
   EXPECT_EQ(*b1, 10u);
   EXPECT_EQ(*b2, std::vector<int>(20, 42));
   EXPECT_EQ(count, 20u);
}

TEST_F(TDFRanges, NoEarlyStopping)
{
   auto &d = GetTDF();
   auto f = d.Filter([](int b) { return b % 2 == 0; }, {"tdfentry_"});
   auto b3 = f.Range(2).Count();
   auto b4 = f.Count();
}

#ifdef R__USE_IMT
TEST(TDFRangesMT, ThrowIfIMT)
{
   bool hasThrown = false;
   ROOT::EnableImplicitMT();
   TDataFrame d(0);
   try {
      d.Range(0);
   } catch (const std::exception &e) {
      hasThrown = true;
      EXPECT_STREQ(e.what(), "Range was called with ImplicitMT enabled. Multi-thread ranges are not supported.");
   }
   EXPECT_TRUE(hasThrown);
}
#endif

/**** REGRESSION TESTS ****/
TEST_F(TDFRanges, CorrectEarlyStop)
{
   // one child ending before the father -- only one stop signal must be propagated upstream
   auto &d = GetTDF();
   auto twenty = d.Range(10, 50).Range(10, 20).Min<ULong64_t>("tdfentry_");
   auto four = d.Filter([](ULong64_t b) { return b > 95; }, {"tdfentry_"}).Range(10).Count();
   EXPECT_EQ(*twenty, 20u);
   EXPECT_EQ(*four, 4u);

   // child and parent ending on the same entry -- only one stop signal must be propagated upstream
   auto two = d.Range(2).Range(2).Count();
   auto ten = d.Range(10).Count();
   EXPECT_EQ(*two, 2u);
   EXPECT_EQ(*ten, 10u);
}

TEST_F(TDFRanges, FinishAllActions)
{
   // regression test for ROOT-9232
   // reaching stop with multiple actions to be processed, remaining actions must be processed for this last entry
   auto &d = GetTDF();
   auto ranged = d.Range(0, 3);
   auto c1 = ranged.Count();
   auto c2 = ranged.Count();
   EXPECT_EQ(*c1, 3ull);
   EXPECT_EQ(*c2, *c1);
}

TEST_F(TDFRanges, EntryLoss)
{
   // regression test for ROOT-9272
   auto d = GetTDF();
   auto d_0_30 = d.Range(0, 30);
   EXPECT_EQ(*d_0_30.Count(), 30u);
   EXPECT_EQ(*d_0_30.Count(), 30u);
}
/****** END REGRESSION TESTS ******/
