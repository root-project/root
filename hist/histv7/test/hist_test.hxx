#ifndef hist_test
#define hist_test

#include <ROOT/RAxes.hxx>
#include <ROOT/RBinIndex.hxx>
#include <ROOT/RBinIndexRange.hxx>
#include <ROOT/RBinWithError.hxx>
#include <ROOT/RCategoricalAxis.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistAutoAxisFiller.hxx>
#include <ROOT/RHistEngine.hxx>
#include <ROOT/RHistStats.hxx>
#include <ROOT/RRegularAxis.hxx>
#include <ROOT/RVariableBinAxis.hxx>
#include <ROOT/RWeight.hxx>

using ROOT::Experimental::RAxisVariant;
using ROOT::Experimental::RBinIndex;
using ROOT::Experimental::RBinIndexRange;
using ROOT::Experimental::RBinWithError;
using ROOT::Experimental::RCategoricalAxis;
using ROOT::Experimental::RHist;
using ROOT::Experimental::RHistAutoAxisFiller;
using ROOT::Experimental::RHistEngine;
using ROOT::Experimental::RHistStats;
using ROOT::Experimental::RRegularAxis;
using ROOT::Experimental::RVariableBinAxis;
using ROOT::Experimental::RWeight;
using ROOT::Experimental::Internal::RAxes;

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

template <typename Work>
void StressInParallel(std::size_t nThreads, Work &&w)
{
   std::atomic<bool> flag;

   std::vector<std::thread> threads;
   for (std::size_t i = 0; i < nThreads; i++) {
      threads.emplace_back([&] {
         while (!flag) {
            // Wait for all threads to be started.
         }
         w();
      });
   }

   flag = true;
   for (std::size_t i = 0; i < nThreads; i++) {
      threads[i].join();
   }
}

#endif
