#include "ROOT/TDataFrame.hxx"
#include "TRandom.h"
#include "gtest/gtest.h"
#include <limits>
using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;
using namespace ROOT::Detail::TDF;

/********* FIXTURES *********/
// fixture that provides a TDF with no data-source and a single column "x" containing normal-distributed doubles
class TDFCallbacks : public ::testing::Test {
protected:
   const ULong64_t nEvents = 8ull; // must be initialized before fLoopManager

private:
   TDataFrame fLoopManager;
   TInterface<TLoopManager> DefineRandomCol()
   {
      TRandom r;
      return fLoopManager.Define("x", [r]() mutable { return r.Gaus(); });
   }

protected:
   TDFCallbacks() : fLoopManager(nEvents), tdf(DefineRandomCol()) {}
   TInterface<TLoopManager> tdf;
};

/********* TESTS *********/
TEST_F(TDFCallbacks, Histo1DWithFillTOHelper)
{
   // Histo1D<double> + OnPartialResult + FillTOHelper
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.OnPartialResult(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), everyN * i);
   });
   *h;
   EXPECT_EQ(nEvents, everyN * i);
}

TEST_F(TDFCallbacks, JittedHisto1DWithFillTOHelper)
{
   // Histo1D + Jitting + OnPartialResult + FillTOHelper
   auto h = tdf.Histo1D({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.OnPartialResult(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), everyN * i);
   });
   *h;
   EXPECT_EQ(nEvents, everyN * i);
}

TEST_F(TDFCallbacks, Histo1DWithFillHelper)
{
   // Histo1D<double> + OnPartialResult + FillHelper
   auto h = tdf.Histo1D<double>("x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.OnPartialResult(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), everyN * i);
   });
   *h;
   EXPECT_EQ(nEvents, everyN * i);
}

TEST_F(TDFCallbacks, JittedHisto1DWithFillHelper)
{
   // Histo1D + Jitting + OnPartialResult + FillHelper
   auto h = tdf.Histo1D("x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.OnPartialResult(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), everyN * i);
   });
   *h;
   EXPECT_EQ(nEvents, everyN * i);
}

TEST_F(TDFCallbacks, Min)
{
   // Min + OnPartialResult
   auto m = tdf.Min<double>("x");
   double runningMin = std::numeric_limits<double>::max();
   m.OnPartialResult(2, [&runningMin](double x) {
      EXPECT_LE(x, runningMin);
      runningMin = x;
   });
   EXPECT_DOUBLE_EQ(runningMin, *m);
}

TEST_F(TDFCallbacks, JittedMin)
{
   // Min + Jitting + OnPartialResult
   auto m = tdf.Min("x");
   double runningMin = std::numeric_limits<double>::max();
   m.OnPartialResult(2, [&runningMin](double x) {
      EXPECT_LE(x, runningMin);
      runningMin = x;
   });
   EXPECT_DOUBLE_EQ(runningMin, *m);
}

TEST_F(TDFCallbacks, Max)
{
   // Max + OnPartialResult
   auto m = tdf.Max<double>("x");
   double runningMax = std::numeric_limits<double>::lowest();
   m.OnPartialResult(2, [&runningMax](double x) {
      EXPECT_GE(x, runningMax);
      runningMax = x;
   });
   EXPECT_DOUBLE_EQ(runningMax, *m);
}

TEST_F(TDFCallbacks, JittedMax)
{
   // Max + Jitting + OnPartialResult
   auto m = tdf.Max("x");
   double runningMax = std::numeric_limits<double>::lowest();
   m.OnPartialResult(2, [&runningMax](double x) {
      EXPECT_GE(x, runningMax);
      runningMax = x;
   });
   EXPECT_DOUBLE_EQ(runningMax, *m);
}

TEST_F(TDFCallbacks, Mean)
{
   // Mean + OnPartialResult
   auto m = tdf.Mean<double>("x");
   // TODO find a better way to check that the running mean makes sense
   bool called = false;
   m.OnPartialResult(nEvents / 2, [&called](double) { called = true; });
   *m;
   EXPECT_TRUE(called);
}

TEST_F(TDFCallbacks, JittedMean)
{
   // Mean + Jitting + OnPartialResult
   auto m = tdf.Mean("x");
   // TODO find a better way to check that the running mean makes sense
   bool called = false;
   m.OnPartialResult(nEvents / 2, [&called](double) { called = true; });
   *m;
   EXPECT_TRUE(called);
}

TEST_F(TDFCallbacks, MultipleCallbacks)
{
   // registration of multiple callbacks on the same partial result
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t everyN = 1ull;
   ULong64_t i = 0ull;
   h.OnPartialResult(everyN, [&i, everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), i);
   });

   everyN = 2ull;
   ULong64_t i2 = 0ull;
   h.OnPartialResult(everyN, [&i2, everyN](value_t &h_) {
      i2 += everyN;
      EXPECT_EQ(h_.GetEntries(), i2);
   });

   *h;
}

TEST_F(TDFCallbacks, MultipleEventLoops)
{
   // callbacks must be de-registered after the event-loop is run
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   h.OnPartialResult(1ull, [&i](value_t &) { ++i; });
   *h;

   auto h2 = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   *h2;

   EXPECT_EQ(i, nEvents);
}
