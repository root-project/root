#define R__USE_IMT
#include "ThreadPool.h"
#include <vector>
#include <chrono>
#include <iostream>
#include "tbb/tbb.h"
#include "tbb/parallel_for.h"

int main()
{
   size_t n = 100000000;
   std::vector<float> data1(n);
   std::vector<float> data2(n);

   for (size_t i = 0; i < 100000000; i++) {
      data1[i] = i;
   }

   float * __restrict__ d1 = data1.data();
   float * __restrict__ d2 = data2.data();

   auto f_range = [&](const tbb::blocked_range<size_t> & x)
   {
      size_t a = x.begin();
      size_t b = x.end();
      for (size_t i = a; i < b; i++) {
         d2[i] = 2.0 * d1[i] * d1[i];
      }
     return 1.0;
   };

   auto f = [&](size_t i)
   {
      d2[i] = 2.0 * d1[i] * d1[i];
      return 1.0;
   };

   ThreadPool pool{};
   auto start = std::chrono::steady_clock::now();
   for (size_t i = 0; i < 10; i++) {
      pool.Map(f, ROOT::TSeqI(n));
   }
   auto end = std::chrono::steady_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
   std::cout << "Time using ThreadPool: " << duration.count() << std::endl;

   start = std::chrono::steady_clock::now();
   for (size_t i = 0; i < 10; i++) {
      parallel_for(tbb::blocked_range<size_t>(0, n), f_range);
   }
   end = std::chrono::steady_clock::now();
   duration = std::chrono::duration_cast<std::chrono::milliseconds>(end -start);
   std::cout << "Time using tbb:        " << duration.count() << std::endl;
}
