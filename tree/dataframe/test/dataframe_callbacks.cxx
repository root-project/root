#include "ROOT/RDataFrame.hxx"
#include "TRandom.h"
#include "TROOT.h"
#include "gtest/gtest.h"
#include <limits>
;
using namespace ROOT::RDF;
using namespace ROOT::Detail::RDF;

/********* FIXTURES *********/
static constexpr ULong64_t gNEvents = 8ull;

// fixture that provides a RDF with no data-source and a single column "x" containing normal-distributed doubles
class RDFCallbacks : public ::testing::Test {
private:
   ROOT::RDataFrame fLoopManager;
   RInterface<RLoopManager> DefineRandomCol()
   {
      TRandom r;
      return fLoopManager.Define("x", [r]() mutable { return r.Gaus(); });
   }

protected:
   RDFCallbacks() : fLoopManager(gNEvents), tdf(DefineRandomCol()) {}
   RInterface<RLoopManager> tdf;
};

#ifdef R__USE_IMT
static constexpr unsigned int gNSlots = 4u;

// fixture that enables implicit MT and provides a RDF with no data-source and a single column "x" containing
// normal-distributed doubles
class RDFCallbacksMT : public ::testing::Test {
   class TIMTEnabler {
   public:
      TIMTEnabler(unsigned int sl) { ROOT::EnableImplicitMT(sl); }
      ~TIMTEnabler() { ROOT::DisableImplicitMT(); }
   };

private:
   TIMTEnabler fIMTEnabler;
   ROOT::RDataFrame fLoopManager;
   RInterface<RLoopManager> DefineRandomCol()
   {
      std::vector<TRandom> rs(gNSlots);
      return fLoopManager.DefineSlot("x", [rs](unsigned int slot) mutable { return rs[slot].Gaus(); });
   }

protected:
   RDFCallbacksMT() : fIMTEnabler(gNSlots), fLoopManager(gNEvents), tdf(DefineRandomCol()) {}
   RInterface<RLoopManager> tdf;
};
#endif

/********* TESTS *********/
TEST_F(RDFCallbacks, Histo1DWithFillTOHelper)
{
   // Histo1D<double> + OnPartialResult + FillTOHelper
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.OnPartialResult(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), i);
   });
   *h;
   EXPECT_EQ(gNEvents, i);
}

TEST_F(RDFCallbacks, JittedHisto1DWithFillTOHelper)
{
   // Histo1D + Jitting + OnPartialResult + FillTOHelper
   auto h = tdf.Histo1D({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.OnPartialResult(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), i);
   });
   *h;
   EXPECT_EQ(gNEvents, i);
}

TEST_F(RDFCallbacks, Histo1DWithFillHelper)
{
   // Histo1D<double> + OnPartialResult + FillHelper
   auto h = tdf.Histo1D<double>("x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.OnPartialResult(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), i);
   });
   *h;
   EXPECT_EQ(gNEvents, i);
}

TEST_F(RDFCallbacks, JittedHisto1DWithFillHelper)
{
   // Histo1D + Jitting + OnPartialResult + FillHelper
   auto h = tdf.Histo1D("x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   ULong64_t everyN = 1ull;
   h.OnPartialResult(everyN, [&i, &everyN](value_t &h_) {
      i += everyN;
      EXPECT_EQ(h_.GetEntries(), i);
   });
   *h;
   EXPECT_EQ(gNEvents, i);
}

TEST_F(RDFCallbacks, Min)
{
   // Min + OnPartialResult
   auto m = tdf.Min<double>("x");
   double runningMin = std::numeric_limits<double>::max();
   m.OnPartialResult(2, [&runningMin](double x) {
      EXPECT_LE(x, runningMin);
      runningMin = x;
   });
   *m;
   EXPECT_DOUBLE_EQ(runningMin, *m);
}

TEST_F(RDFCallbacks, JittedMin)
{
   // Min + Jitting + OnPartialResult
   auto m = tdf.Min("x");
   double runningMin = std::numeric_limits<double>::max();
   m.OnPartialResult(2, [&runningMin](double x) {
      EXPECT_LE(x, runningMin);
      runningMin = x;
   });
   *m;
   EXPECT_DOUBLE_EQ(runningMin, *m);
}

TEST_F(RDFCallbacks, Max)
{
   // Max + OnPartialResult
   auto m = tdf.Max<double>("x");
   double runningMax = std::numeric_limits<double>::lowest();
   m.OnPartialResult(2, [&runningMax](double x) {
      EXPECT_GE(x, runningMax);
      runningMax = x;
   });
   *m;
   EXPECT_DOUBLE_EQ(runningMax, *m);
}

TEST_F(RDFCallbacks, JittedMax)
{
   // Max + Jitting + OnPartialResult
   auto m = tdf.Max("x");
   double runningMax = std::numeric_limits<double>::lowest();
   m.OnPartialResult(2, [&runningMax](double x) {
      EXPECT_GE(x, runningMax);
      runningMax = x;
   });
   *m;
   EXPECT_DOUBLE_EQ(runningMax, *m);
}

TEST_F(RDFCallbacks, Mean)
{
   // Mean + OnPartialResult
   auto m = tdf.Mean<double>("x");
   // TODO find a better way to check that the running mean makes sense
   bool called = false;
   m.OnPartialResult(gNEvents / 2, [&called](double) { called = true; });
   *m;
   EXPECT_TRUE(called);
}

TEST_F(RDFCallbacks, JittedMean)
{
   // Mean + Jitting + OnPartialResult
   auto m = tdf.Mean("x");
   // TODO find a better way to check that the running mean makes sense
   bool called = false;
   m.OnPartialResult(gNEvents / 2, [&called](double) { called = true; });
   *m;
   EXPECT_TRUE(called);
}

TEST_F(RDFCallbacks, Take)
{
   // Take + OnPartialResult
   auto t = tdf.Take<double>("x");
   unsigned int i = 0u;
   t.OnPartialResult(1, [&](decltype(t)::Value_t &t_) {
      ++i;
      EXPECT_EQ(t_.size(), i);
   });
   *t;
}

TEST_F(RDFCallbacks, Count)
{
   // Count + OnPartialResult
   auto c = tdf.Count();
   ULong64_t i = 0ull;
   c.OnPartialResult(1, [&](decltype(c)::Value_t c_) {
      ++i;
      EXPECT_EQ(c_, i);
   });
   *c;
   EXPECT_EQ(*c, i);
}

TEST_F(RDFCallbacks, Reduce)
{
   // Reduce + OnPartialResult
   double runningMin;
   auto m = tdf.Min<double>("x").OnPartialResult(1, [&runningMin](double m_) { runningMin = m_; });
   auto r = tdf.Reduce([](double x1, double x2) { return std::min(x1, x2); }, {"x"}, 0.);
   r.OnPartialResult(1, [&runningMin](double r_) { EXPECT_DOUBLE_EQ(r_, runningMin); });
   *r;
}

TEST_F(RDFCallbacks, Chaining)
{
   // Chaining of multiple OnPartialResult[Slot] calls
   unsigned int i = 0u;
   auto c = tdf.Count()
               .OnPartialResult(1, [&i](ULong64_t) { ++i; })
               .OnPartialResultSlot(1, [&i](unsigned int, ULong64_t) {++i; });
   *c;
   EXPECT_EQ(i, gNEvents * 2);
}

TEST_F(RDFCallbacks, OrderOfExecution)
{
   // Test that callbacks are executed in the order they are registered
   unsigned int i = 0u;
   auto c = tdf.Count();
   c.OnPartialResult(1, [&i](ULong64_t) {
      EXPECT_EQ(i, 0u);
      i = 42u;
   });
   c.OnPartialResultSlot(1, [&i](unsigned int slot, ULong64_t) {
      if (slot == 0u) {
         EXPECT_EQ(i, 42u);
         i = 0u;
      }
   });
   *c;
   EXPECT_EQ(i, 0u);
}

TEST_F(RDFCallbacks, ExecuteOnce)
{
   // OnPartialResult(kOnce)
   auto c = tdf.Count();
   unsigned int callCount = 0;
   c.OnPartialResult(0, [&callCount](ULong64_t) { callCount++; });
   *c;
   EXPECT_EQ(callCount, 1u);
}

TEST_F(RDFCallbacks, MultipleCallbacks)
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

TEST_F(RDFCallbacks, MultipleEventLoops)
{
   // callbacks must be de-registered after the event-loop is run
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   ULong64_t i = 0ull;
   h.OnPartialResult(1ull, [&i](value_t &) { ++i; });
   *h;

   auto h2 = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   *h2;

   EXPECT_EQ(i, gNEvents);
}

class FunctorClass {
   unsigned int &i_;

public:
   FunctorClass(unsigned int &i) : i_(i) {}
   void operator()(ULong64_t) { ++i_; }
};

TEST_F(RDFCallbacks, FunctorClass)
{
   unsigned int i = 0;
   *(tdf.Count().OnPartialResult(1, FunctorClass(i)));
   EXPECT_EQ(i, gNEvents);
}

unsigned int freeFunctionCounter = 0;
void FreeFunction(ULong64_t)
{
   freeFunctionCounter++;
}

TEST_F(RDFCallbacks, FreeFunction)
{
   *(tdf.Count().OnPartialResult(1, FreeFunction));
   EXPECT_EQ(freeFunctionCounter, gNEvents);
}


#ifdef R__USE_IMT
/******** Multi-thread tests **********/
TEST_F(RDFCallbacksMT, ExecuteOncePerSlot)
{
   // OnPartialResultSlot(kOnce)
   auto c = tdf.Count();
   std::atomic_uint callCount(0u);
   c.OnPartialResultSlot(c.kOnce, [&callCount](unsigned int, ULong64_t) { callCount++; });
   *c;
   // depending on how tasks are dispatched to worker threads and how quickly threads push and pop slot numbers from
   // TSlotStack, the callback might be executed 1 to nSlots times.
   EXPECT_LE(callCount, gNSlots);
   EXPECT_GT(callCount, 0u);
}

TEST_F(RDFCallbacksMT, ExecuteOnce)
{
   // OnPartialResult(kOnce)
   auto c = tdf.Count();
   std::atomic_uint callCount(0u);
   c.OnPartialResult(c.kOnce, [&callCount](ULong64_t) { callCount++; });
   *c;
   EXPECT_EQ(callCount, 1u);
}

TEST_F(RDFCallbacksMT, Histo1DWithFillTOHelper)
{
   // Histo1D<double> + OnPartialResultSlot + FillTOHelper
   auto h = tdf.Histo1D<double>({"", "", 128, -2., 2.}, "x");
   using value_t = typename decltype(h)::Value_t;
   std::array<ULong64_t, gNSlots> is;
   is.fill(0ull);
   constexpr ULong64_t everyN = 1ull;
   h.OnPartialResultSlot(everyN, [&](unsigned int slot, value_t &h_) {
      is[slot] += everyN;
      EXPECT_EQ(h_.GetEntries(), is[slot]);
   });
   *h;
   EXPECT_EQ(gNEvents, std::accumulate(is.begin(), is.end(), 0ull));
}

TEST_F(RDFCallbacksMT, Histo1DWithFillHelper)
{
   // Histo1D<double> + OnPartialResultSlot + FillHelper
   auto h = tdf.Histo1D<double>("x");
   using value_t = typename decltype(h)::Value_t;
   std::array<ULong64_t, gNSlots> is;
   is.fill(0ull);
   constexpr ULong64_t everyN = 1ull;
   h.OnPartialResultSlot(everyN, [&](unsigned int slot, value_t &h_) {
      is[slot] += everyN;
      EXPECT_EQ(h_.GetEntries(), is[slot]);
   });
   *h;
   EXPECT_EQ(gNEvents, std::accumulate(is.begin(), is.end(), 0ull));
}

TEST(RDFCallbacksMTMore, LessTasksThanWorkers)
{
   ROOT::EnableImplicitMT(4);
   ROOT::RDataFrame d(1);
   auto c = d.Count();
   std::atomic_uint counter(0u);
   c.OnPartialResult(c.kOnce, [&counter](ULong64_t) { counter++; });
   *c;
   EXPECT_EQ(counter, 1u);

   ROOT::DisableImplicitMT();
}

#endif
