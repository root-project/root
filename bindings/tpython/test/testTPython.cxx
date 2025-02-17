#include <TPython.h>

#include <any>
#include <iostream>
#include <sstream>
#include <thread>

#include "gtest/gtest.h"

namespace {

void task1(int i)
{
   std::stringstream code;
   code << "arr.append(" << i << ")";
   TPython::Exec(code.str().c_str());
}

} // namespace

// Test TPython::Exec from multiple threads.
TEST(TPython, ExecMultithreading)
{
   int nThreads = 100;

   // Concurrently append to this array
   TPython::Exec("arr = []");

   std::vector<std::thread> threads;
   for (int i = 0; i < nThreads; i++) {
      threads.emplace_back(task1, i);
   }

   for (decltype(threads.size()) i = 0; i < threads.size(); i++) {
      threads[i].join();
   }

   // In the end, let's check if the size is correct.
   std::any len;
   TPython::Exec("_anyresult = ROOT.std.make_any['int'](len(arr))", &len);
   EXPECT_EQ(std::any_cast<int>(len), nThreads);
}
