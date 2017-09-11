#include "ROOT/TDFNodes.hxx"

#include <thread>

#include "gtest/gtest.h"

using namespace ROOT::Experimental;

TEST(TDataFrameNodes, TSlotStackCheckSameThreadSameSlot)
{
   unsigned int n(7);
   ROOT::Internal::TDF::TSlotStack s(n);
   EXPECT_EQ(s.GetSlot(), s.GetSlot());
}

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
   auto theTest = []() {
      unsigned int n(2);
      ROOT::Internal::TDF::TSlotStack s(n);

      std::vector<std::thread> ts;

      for (unsigned int i = 0; i < 2; ++i) {
         ts.emplace_back([&s]() { s.GetSlot(); });
      }
      for (unsigned int i = 0; i < 2; ++i) {
         ts.emplace_back([&s, i]() { s.ReturnSlot(i); });
      }

      for (auto &&t : ts)
         t.join();
   };

   ASSERT_DEATH(theTest(), "TSlotStack has a reference count relative to an index which will become negative");
}
