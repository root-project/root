/// \file
/// \ingroup tutorial_histv7
///
/// Filling RHist using RDataFrame.
///
/// \macro_code
/// \macro_output
/// \macro_image
///
/// \date February 2026
/// \author The ROOT Team

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/Hist/ConvertToTH1.hxx>
#include <TCanvas.h>
#include <TH1.h>

void hist101_RDataFrame_Hist()
{
   // Create a data frame with 100 rows.
   ROOT::RDataFrame df(100);

   // Define a new column that will be used to fill the histogram.
   auto dfX = df.Define("x",
                        [](ULong64_t e) {
                           double v = std::exp(0.1 * e);
                           return std::fmod(v, 20.0);
                        },
                        {"rdfentry_"});

   // Create the histogram with 40 bins from 0.0 to 20.0.
   auto hist = dfX.Hist(40, {0.0, 20.0}, "x");

   // Print (some of) the global statistics.
   std::cout << "entries = " << hist->GetNEntries();
   std::cout << ", mean = " << hist->ComputeMean();
   std::cout << ", stddev = " << hist->ComputeStdDev();
   std::cout << "\n";

   // Convert the histogram to TH1D.
   auto h1 = ROOT::Experimental::Hist::ConvertToTH1D(*hist);

   // Draw the converted TH1D.
   auto *c = new TCanvas("c", "", 10, 10, 900, 500);
   h1->DrawCopy("", "x");
}
