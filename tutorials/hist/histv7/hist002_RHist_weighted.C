/// \file
/// \ingroup tutorial_histv7
///
/// Weighted filling of RHist and RBinWithError bin content type.
///
/// \macro_code
/// \macro_output
///
/// \date October 2025
/// \author The ROOT Team

#include <ROOT/RBinIndex.hxx>
#include <ROOT/RBinWithError.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RRegularAxis.hxx>
#include <ROOT/RWeight.hxx>

#include <cstddef>
#include <iostream>
#include <random>
#include <variant>

// It is currently not possible to directly draw an RHist, so this function implements an output with ASCII characters.
static void DrawHistogram(const ROOT::Experimental::RHist<double> &hist)
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
         static constexpr int Scale = 100;
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

void hist002_RHist_weighted()
{
   // Create an axis that can be used for multiple histograms.
   ROOT::Experimental::RRegularAxis axis(40, {0.0, 20.0});

   // Create two histograms, one of which will be filled with weighted entries.
   ROOT::Experimental::RHist<double> hist1({axis});
   ROOT::Experimental::RHist<double> hist2({axis});

   // Create a normal distribution with mean 10.0 and stddev 5.0.
   std::mt19937 gen;
   std::normal_distribution normal(10.0, 5.0);
   for (std::size_t i = 0; i < 25000; i++) {
      hist1.Fill(normal(gen));
      double value = normal(gen);
      double weight = 0.2 + 0.008 * value * value;
      hist2.Fill(value, ROOT::Experimental::RWeight(weight));
   }

   // "Draw" the histograms with ASCII characters.
   std::cout << "hist1 with expected mean = " << normal.mean() << "\n";
   DrawHistogram(hist1);
   std::cout << "\n";

   std::cout << "hist2 with distorted normal distribution\n";
   DrawHistogram(hist2);
   std::cout << "\n";

   // Create and fill a third histogram with the special RBinWithError bin content type.
   // In addition to the sum of weights, it tracks the sum of weights squared to compute the bin errors.
   ROOT::Experimental::RHist<ROOT::Experimental::RBinWithError> hist3({axis});
   for (std::size_t i = 0; i < 25000; i++) {
      double value = normal(gen);
      double weight = 0.2 + 0.008 * value * value;
      hist3.Fill(value, ROOT::Experimental::RWeight(weight));
   }

   // Visualize the computed bin errors with ASCII characters.
   std::cout << "bin errors of hist3 (not to scale)\n";
   for (int row = 15; row > 0; row--) {
      auto print = [&](ROOT::Experimental::RBinIndex bin) {
         auto error = std::sqrt(hist3.GetBinContent(bin).fSum2);
         static constexpr int Scale = 5;
         std::cout << (error >= (row * Scale) ? '*' : ' ');
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
