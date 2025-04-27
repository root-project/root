#include "ROOT/TestSupport.hxx"

#include <ROOT/RDataFrame.hxx>
#include <TStatistic.h> // To check reading of columns with types which are mothers of the column type
#include <TSystem.h>

#include <chrono>
#include <thread>
#include <set>
#include <stdexcept> // std::runtime_error

#include "gtest/gtest.h"

#if defined(R__USE_IMT) && !defined(NDEBUG)
#include <ROOT/RSlotStack.hxx>

TEST(RDataFrameNodes, RSlotStackGetOneTooMuch)
{
   using namespace std::chrono_literals;

   constexpr unsigned int NSlot = 2;
   ROOT::Internal::RSlotStack s(NSlot);
   std::vector<std::thread> ts;

   for (unsigned int i = 0; i < NSlot + 1; ++i) {
      ts.emplace_back([&s, NSlot]() {
         const auto slot = s.GetSlot();
         EXPECT_LT(slot, NSlot);
         std::this_thread::sleep_for(10ms);
         s.ReturnSlot(slot);
      });
   }

   for (auto &&t : ts)
      t.join();
}

TEST(RDataFrameNodes, RSlotStackPutBackTooMany)
{
   auto theTest = []() {
      ROOT::Internal::RSlotStack s(1);
      s.ReturnSlot(0);
   };

   EXPECT_THROW(theTest(), std::logic_error);
}

// Run with 16 threads with 8 slots, and ensure that slot numbers
// are always unique.
TEST(RDataFrameNodes, RSlotStackUnique)
{
   constexpr unsigned int N = 8;
   ROOT::Internal::RSlotStack s(N);
   std::set<unsigned int> slots;
   std::mutex mutex;

   auto insert = [&](unsigned int slot, unsigned int threadId) {
      bool inserted = false;
      {
         std::scoped_lock lock{mutex};
         inserted = slots.insert(slot).second;
      }
      EXPECT_TRUE(inserted) << "Slot " << slot << " of thread " << threadId << " is already taken.";
   };
   auto remove = [&](unsigned int slot, unsigned int threadId) {
      unsigned int nErased = 0;
      {
         std::scoped_lock lock{mutex};
         nErased = slots.erase(slot);
      }
      EXPECT_EQ(nErased, 1) << "Slot " << slot << " of thread " << threadId << " doesn't seem to be assigned.";
   };

   auto slotTask = [&](unsigned int threadId) {
      using namespace std::chrono_literals;
      ROOT::Internal::RSlotStackRAII slot{s};
      insert(slot.fSlot, threadId);

      std::this_thread::sleep_for(threadId * 1us);

      remove(slot.fSlot, threadId);
   };

   auto runThreads = [&slotTask, &N]() {
      std::vector<std::thread> ts;
      for (unsigned int i = 0; i < 2 * N; ++i) {
         ts.emplace_back(slotTask, i);
      }
      for (auto &&t : ts)
         t.join();
   };

   runThreads();
   ASSERT_TRUE(slots.empty());
   runThreads();
   ASSERT_TRUE(slots.empty());
}
#endif

TEST(RDataFrameNodes, RLoopManagerGetLoopManagerUnchecked)
{
   ROOT::Detail::RDF::RLoopManager lm{};
   ASSERT_EQ(&lm, lm.GetLoopManagerUnchecked());
}

TEST(RDataFrameNodes, RLoopManagerJitWrongCode)
{
   ROOT::Detail::RDF::RLoopManager lm{};
   lm.ToJitExec("souble d = 3.14");
   EXPECT_THROW(lm.Run(), std::runtime_error) << "Bogus C++ code was jitted and nothing was detected!";
}

TEST(RDataFrameNodes, DoubleEvtLoop)
{
   ROOT::RDataFrame d1(4);
   auto d = d1.Define("x", []() { return 2; });

   std::vector<std::string> files{"f1.root", "f2.root"};

   for (auto &f : files)
      d.Snapshot<int>("t1", f, {"x"});

   ROOT::RDataFrame tdf("t1", files);
   *tdf.Count();

   // Check that this is not printed
   // Warning in <TTreeReader::SetEntryBase()>: The current tree in the TChain t1 has changed (e.g. by TTree::Process)
   // even though TTreeReader::SetEntry() was called, which switched the tree again. Did you mean to call
   // TTreeReader::SetLocalEntry()?

   ROOT_EXPECT_NODIAG(*tdf.Count());

   for (auto &f : files)
      gSystem->Unlink(f.c_str());
}

// ROOT-9736
TEST(RDataFrameNodes, InheritanceOfDefines)
{
   ROOT::RDataFrame df(1);
   const auto nBinsExpected = 42;
   // Read the TH1F as a TH1
   df.Define("b", [&]() { return TH1F("b", "b", nBinsExpected, 0, 1); })
      .Foreach([&](TH1 &h) { EXPECT_EQ(h.GetNbinsX(), nBinsExpected);}, {"b"});

   const auto ofileName = "InheritanceOfDefines.root";

   const auto val = 42.;
   auto createStat = [&val]() {
      TStatistic t;
      t.Fill(val);
      return t;
   };

   // Read as TObject from disk a TStatistics object
   auto checkStat = [&val](TObject &o) { EXPECT_EQ(val, ((TStatistic *)&o)->GetMean()); };
   ROOT::RDataFrame(1).Define("x", createStat).Snapshot<TStatistic>("t", ofileName, {"x"})->Foreach(checkStat, {"x"});
   gSystem->Unlink(ofileName);
}

TEST(RDataFrameNodes, InvalidLoopType)
{
   ROOT::Detail::RDF::RLoopManager lm{};
   EXPECT_THROW(lm.Run(), std::runtime_error) << "An invalid event loop was run!";
}
