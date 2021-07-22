#include <ROOT/RWorkQueue.hxx>

#include <thread>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

template <class T>
using RWorkQueue = ROOT::Internal::RWorkQueue<T>;

TEST(RWorkQueue, Basics)
{
   RWorkQueue<int> wq(4);
   EXPECT_TRUE(wq.IsEmpty());
   EXPECT_FALSE(wq.IsFull());

   wq.Enqueue(0);
   EXPECT_FALSE(wq.IsEmpty());
   EXPECT_FALSE(wq.IsFull());
   wq.Enqueue(1);
   wq.Enqueue(2);
   wq.Enqueue(3);
   EXPECT_TRUE(wq.IsFull());

   std::vector<int> result;
   std::thread producer([&]() {
      for (int i = 4; i < 100000; ++i)
         wq.Enqueue(i+0);
      wq.Enqueue(-1);
   });
   std::thread consumer([&]() {
      while (true) {
         auto x = wq.Pop();
         if (x == -1)
            break;
         result.emplace_back(x);
      }
   });

   producer.join();
   consumer.join();

   EXPECT_TRUE(wq.IsEmpty());
   EXPECT_EQ(100000u, result.size());
   for (unsigned int i = 0; i < result.size(); ++i)
      EXPECT_EQ(i, result[i]);
}
