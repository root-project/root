#include <ROOT/TDataFrame.hxx>
#include <ROOT/TDFNodes.hxx>
#include <TSystem.h>

#include <mutex>
#include <thread>

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

TEST(TDataFrameNodes, TSlotStackCheckSameThreadSameSlot)
{
   unsigned int n(7);
   ROOT::Internal::TDF::TSlotStack s(n);
   EXPECT_EQ(s.GetSlot(), s.GetSlot());
}

#ifndef NDEBUG

TEST(TDataFrameNodes, TSlotStackGetOneTooMuch)
{
   auto theTest = []() {
      unsigned int n(2);
      ROOT::Internal::TDF::TSlotStack s(n);

      std::vector<std::thread> ts;

      for (unsigned int i = 0; i < 3; ++i) {
         ts.emplace_back([&s]() { s.GetSlot(); });
      }

      for (auto &&t : ts)
         t.join();
   };

   ASSERT_DEATH(theTest(), "TSlotStack assumes that a value can be always obtained.");
}

TEST(TDataFrameNodes, TSlotStackPutBackTooMany)
{
   std::mutex m;
   auto theTest = [&m]() {
      unsigned int n(2);
      ROOT::Internal::TDF::TSlotStack s(n);

      std::vector<std::thread> ts;

      for (unsigned int i = 0; i < 2; ++i) {
         ts.emplace_back([&s, &m]() {
            std::lock_guard<std::mutex> lg(m);
            s.GetSlot();
         });
      }
      for (unsigned int i = 0; i < 2; ++i) {
         ts.emplace_back([&s, &m, i]() {
            std::lock_guard<std::mutex> lg(m);
            s.ReturnSlot(i);
         });
      }

      for (auto &&t : ts)
         t.join();
   };

   ASSERT_DEATH(theTest(), "TSlotStack has a reference count relative to an index which will become negative");
}

#endif

TEST(TDataFrameNodes, TLoopManagerGetImplPtr)
{
   ROOT::Detail::TDF::TLoopManager lm(nullptr, {});
   ASSERT_EQ(&lm, lm.GetImplPtr());
}

TEST(TDataFrameNodes, TLoopManagerJit)
{
   ROOT::Detail::TDF::TLoopManager lm(nullptr, {});
   lm.Jit("souble d = 3.14");
   int ret(1);
   try {
      testing::internal::CaptureStderr();
      lm.Run();
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "Bogus C++ code was jitted and nothing was detected!";
}

TEST(TDataFrameNodes, DoubleEvtLoop)
{
   TDataFrame d1(4);
   auto d = d1.Define("x", []() { return 2; });

   std::vector<std::string> files{"f1.root", "f2.root"};

   for (auto &f : files)
      d.Snapshot<int>("t1", f, {"x"});

   TDataFrame tdf("t1", files);
   *tdf.Count();

   // Check that this is not printed
   // Warning in <TTreeReader::SetEntryBase()>: The current tree in the TChain t1 has changed (e.g. by TTree::Process)
   // even though TTreeReader::SetEntry() was called, which switched the tree again. Did you mean to call
   // TTreeReader::SetLocalEntry()?

   testing::internal::CaptureStdout();
   *tdf.Count();
   auto output = testing::internal::GetCapturedStdout();
   EXPECT_STREQ("", output.c_str()) << "An error was printed: " << output << std::endl;

   for (auto &f : files)
      gSystem->Unlink(f.c_str());
}
