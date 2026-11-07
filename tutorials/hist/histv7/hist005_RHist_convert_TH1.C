/// \file
/// \ingroup tutorial_histv7
///
/// Conversion of RHist to TH1.
///
/// \macro_code
/// \macro_output
/// \macro_image
///
/// \date February 2026
/// \author The ROOT Team

#include <ROOT/RHist.hxx>
#include <ROOT/Hist/ConvertToTH1.hxx>
#include <TCanvas.h>
#include <TH1.h>

#include <cstddef>
#include <random>

void hist005_RHist_convert_TH1()
{
   // Create a histogram and fill with random values.
   ROOT::Experimental::RHist<int> hist(40, {0.0, 20.0});

   // Create a normal distribution with mean 10.0 and stddev 4.0.
   std::mt19937 gen;
   std::normal_distribution normal(10.0, 4.0);
   for (std::size_t i = 0; i < 1000; i++) {
      hist.Fill(normal(gen));
   }

   // Convert the histogram to TH1I.
   auto h1 = ROOT::Experimental::Hist::ConvertToTH1I(hist);

   // Print (some of) the global statistics.
   std::cout << "hist with expected mean = " << normal.mean() << "\n";
   std::cout << "entries = " << h1->GetEntries();
   std::cout << ", mean = " << h1->GetMean();
   std::cout << ", stddev = " << h1->GetStdDev();
   std::cout << "\n";

   // Draw the converted TH1I.
   auto *c = new TCanvas("c", "", 10, 10, 900, 500);
   h1->DrawCopy("", "h1");

   // All other known methods of TH1 are also available, such as fitting or statistical tests, but go beyond the scope
   // of this tutorial.
}
