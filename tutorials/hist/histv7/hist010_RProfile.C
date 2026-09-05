/// \file
/// \ingroup tutorial_histv7
///
/// Profile histograms with RProfile.
///
/// \macro_code
/// \macro_output
///
/// \date July 2026
/// \author The ROOT Team

#include <ROOT/RBinIndex.hxx>
#include <ROOT/RProfile.hxx>
#include <ROOT/RRegularAxis.hxx>

#include <cstddef>
#include <iostream>
#include <random>

void hist010_RProfile()
{
   // Create an axis that can be used for multiple histograms.
   ROOT::Experimental::RRegularAxis axis(40, {0.0, 20.0});

   // Create a profile histogram and fill with random values.
   ROOT::Experimental::RProfile profile(40, {0.0, 20.0});

   std::mt19937 gen;
   // Create a first normal distribution with mean 10.0 and stddev 4.0.
   std::normal_distribution normal1(10.0, 4.0);
   // Create a second normal distribution for the value.
   std::normal_distribution normal2(40.0, 10.0);
   for (std::size_t i = 0; i < 1000; i++) {
      double x = normal1(gen);
      double v = normal2(gen);
      profile.Fill(x, v);
   }

   // Print (some of) the global statistics.
   std::cout << "entries = " << profile.GetNEntries() << "\n";
   std::cout << "binned mean = " << profile.ComputeMean(0);
   std::cout << ", stddev = " << profile.ComputeStdDev(0);
   std::cout << "\n";

   // "Draw" the entries of the profile histogram with ASCII characters. The height is hard-coded to work for this
   // tutorial.
   for (int row = 8; row > 0; row--) {
      auto print = [&](ROOT::Experimental::RBinIndex bin) {
         const auto &content = profile.GetBinContent(bin);
         static constexpr int Scale = 10;
         std::cout << (content.fSum >= (row * Scale) ? '*' : ' ');
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

   std::cout << "\n";
   std::cout << "value mean = " << profile.ComputeMean(1);
   std::cout << ", stddev = " << profile.ComputeStdDev(1);
   std::cout << "\n";

   auto printBin = [&](ROOT::Experimental::RBinIndex bin) {
      const auto &content = profile.GetBinContent(bin);
      std::cout << "entries = " << content.fSum << ", mean = " << content.ComputeMean();
   };

   std::cout << "underflow bin: ";
   printBin(ROOT::Experimental::RBinIndex::Underflow());
   std::cout << "\n";

   std::cout << "bin #5: ";
   printBin(5);
   std::cout << "\n";

   std::cout << "bin #10: ";
   printBin(10);
   std::cout << "\n";
}
