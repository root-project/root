/// \file
/// \ingroup tutorial_histv7
///
/// Concurrent filling of RHist.
///
/// \macro_code
/// \macro_output
///
/// \date February 2026
/// \author The ROOT Team

#include <ROOT/RAxisVariant.hxx>
#include <ROOT/RBinIndex.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistConcurrentFiller.hxx>
#include <ROOT/RHistFillContext.hxx>
#include <ROOT/RRegularAxis.hxx>

#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

// It is currently not possible to directly draw an RHist, so this function implements an output with ASCII characters.
static void DrawHistogram(const ROOT::Experimental::RHist<int> &hist)
{
   // Get the axis object from the histogram.
   auto &axis = hist.GetAxes()[0];

   // Print (some of) the global statistics.
   std::cout << "entries = " << hist.GetNEntries();
   std::cout << ", mean = " << hist.ComputeMean();
   std::cout << ", stddev = " << hist.ComputeStdDev();
   std::cout << "\n";

   // "Draw" the histogram with ASCII characters. The height is hard-coded to work for this tutorial.
   for (int row = 15; row > 0; row--) {
      auto print = [&](ROOT::Experimental::RBinIndex bin) {
         auto value = hist.GetBinContent(bin);
         static constexpr int Scale = 15;
         std::cout << (value >= (row * Scale) ? '*' : ' ');
      };

      // First the underflow bin, separated by a vertical bar.
      print(ROOT::Experimental::RBinIndex::Underflow());
      std::cout << '|';

      // Now iterate the normal bins and print a '*' if the value is sufficiently large.
      for (auto bin : axis.GetNormalRange()) {
         print(bin);
      }

      // Finally the overflow bin after a separating vertical bar.
      std::cout << '|';
      print(ROOT::Experimental::RBinIndex::Overflow());
      std::cout << "\n";
   }
}

static void FillHistogram(std::size_t t, ROOT::Experimental::RHistFillContext<int> &context)
{
   // Create a normal distribution with mean 10.0 and stddev 4.0.
   std::mt19937 gen(t);
   std::normal_distribution normal(10.0, 4.0);
   for (std::size_t i = 0; i < 1000; i++) {
      context.Fill(normal(gen));
   }
}

void hist004_RHist_concurrent()
{
   // Create an axis and a histogram that multiple threads will fill concurrently.
   ROOT::Experimental::RRegularAxis axis(40, {0.0, 20.0});
   auto hist = std::make_shared<ROOT::Experimental::RHist<int>>(axis);

   {
      // Fill histogram from multiple threads.
      ROOT::Experimental::RHistConcurrentFiller filler(hist);
      std::vector<std::thread> threads;
      for (std::size_t t = 0; t < 4; t++) {
         threads.emplace_back([t, &filler] {
            // Create a fill context for this thread.
            auto context = filler.CreateFillContext();
            FillHistogram(t, *context);
         });
      }

      for (auto &&t : threads) {
         t.join();
      }
   }

   // "Draw" the histogram with ASCII characters.
   DrawHistogram(*hist);
}
