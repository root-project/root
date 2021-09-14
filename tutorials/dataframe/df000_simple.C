/// \file
/// \ingroup tutorial_dataframe
/// Simple RDataFrame example in C++.
///
/// This tutorial shows a minimal example of RDataFrame that creates an
/// empty-source data frame, adds a new column `x` with random numbers
/// and finally creates and draws a histogram for `x`.
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

   // Create a histogram from `x` and draw it
   auto h = rdf_x.Histo1D("x");
   h->DrawClone();
}
