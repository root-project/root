/// \file
/// \ingroup tutorial_dataframe
/// \notebook -draw
/// Simple RDataFrame example in C++.
///
/// This tutorial shows a minimal example of RDataFrame. It starts without input
/// data, generates a new column `x` with random numbers, and finally draws
/// a histogram for `x`.
///
/// \macro_code
/// \macro_output
///
/// \date September 2021
/// \author Enric Tejedor (CERN)

void df000_simple()
{
   // Create a data frame with 100 rows
   ROOT::RDataFrame rdf(100);

   // Define a new column `x` that contains random numbers
   auto rdf_x = rdf.Define("x", [](){ return gRandom->Rndm(); });

   // Create a histogram from `x`
   auto h = rdf_x.Histo1D("x");

   // At the end of this function, the histogram pointed to by `h` will be deleted.
   // Draw a copy of the histogram object instead:
   h->DrawClone();
}
