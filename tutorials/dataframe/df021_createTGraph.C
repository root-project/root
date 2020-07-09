/// \file
/// \ingroup tutorial_dataframe
/// \notebook -draw
/// \brief Fill a TGraph using RDataFrame.
///
/// \macro_code
/// \macro_image
///
/// \date July 2018
/// \author Enrico Guiraud, Danilo Piparo, Massimo Tumolo



void df021_createTGraph()
{
   ROOT::EnableImplicitMT(2);

   ROOT::RDataFrame d(160);

   // Create a trivial parabola
   auto dd = d.Alias("x", "rdfentry_").Define("y", "x*x");

   auto graph = dd.Graph("x", "y");

   // This tutorial is ran with multithreading enabled. The order in which points are inserted is not known, so to have a meaningful representation points are sorted.
   graph->Sort();
   auto c = new TCanvas();
   graph->DrawClone("APL");
}
