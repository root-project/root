/// \file
/// \ingroup tutorial_histv7
///
/// Basics of RHist, including filling and adding them.
///
/// \macro_code
/// \macro_output
///
/// \date September 2025
/// \author The ROOT Team

#include <ROOT/RBinIndex.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RRegularAxis.hxx>

#include <cstddef>
#include <iostream>
#include <random>
#include <variant>

// It is currently not possible to directly draw an RHist, so this function implements an output with ASCII characters.
static void DrawHistogram(const ROOT::Experimental::RHist<int> &hist)
{
   // Get the axis object from the histogram.
   auto &axis = std::get<ROOT::Experimental::RRegularAxis>(hist.GetAxes()[0]);

   // Print (some of) the global statistics.
   std::cout << "entries = " << hist.GetNEntries();
   std::cout << ", mean = " << hist.ComputeMean();
   std::cout << ", stddev = " << hist.ComputeStdDev();
   std::cout << "\n";

   // "Draw" the histogram with ASCII characters. The height is hard-coded to work for this tutorial.
   for (int row = 15; row > 0; row--) {
      auto print = [&](ROOT::Experimental::RBinIndex bin) {
         auto value = hist.GetBinContent(bin);
         static constexpr int Scale = 10;
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

void hist001_RHist_basics()
{
   // Create an axis that can be used for multiple histograms.
   ROOT::Experimental::RRegularAxis axis(40, {0.0, 20.0});

   // Create a first histogram and fill with random values.
   ROOT::Experimental::RHist<int> hist1({axis});

   // Create a normal distribution with mean 5.0 and stddev 2.0.
   std::mt19937 gen;
   std::normal_distribution normal1(5.0, 2.0);
   for (std::size_t i = 0; i < 1000; i++) {
      hist1.Fill(normal1(gen));
   }

   // "Draw" the histogram with ASCII characters.
   std::cout << "hist1 with expected mean = " << normal1.mean() << "\n";
   DrawHistogram(hist1);
   std::cout << "\n";

   // Create a second histogram and fill with random values of a different distribution.
   ROOT::Experimental::RHist<int> hist2({axis});
   std::normal_distribution normal2(13.0, 4.0);
   for (std::size_t i = 0; i < 1500; i++) {
      hist2.Fill(normal2(gen));
   }
   std::cout << "hist2 with expected mean = " << normal2.mean() << "\n";
   DrawHistogram(hist2);
   std::cout << "\n";

   // Create a third, merged histogram from the two previous two.
   ROOT::Experimental::RHist<int> hist3({axis});
   hist3.Add(hist1);
   hist3.Add(hist2);
   std::cout << "hist3 with expected entries = " << (hist1.GetNEntries() + hist2.GetNEntries()) << "\n";
   DrawHistogram(hist3);
}
