/// \file
/// \ingroup tutorial_dataframe
/// \notebook -draw
/// This tutorial shows how to fill a TGraph using the Dataframe.
///
/// \macro_code
///
/// \date July 2018
/// \author Enrico Guiraud, Danilo Piparo, Massimo Tumolo



void df021_createTGraph()
{
   ROOT::EnableImplicitMT(2);

   const unsigned int NR_ELEMENTS = 160;
   std::vector<int> x(NR_ELEMENTS);
   std::vector<int> y(NR_ELEMENTS);

   for (int i = 0; i < NR_ELEMENTS; ++i){
      y[i] = pow(i,2);
      x[i] = i;
   }

   ROOT::RDataFrame d(NR_ELEMENTS);
   auto dd = d.DefineSlotEntry("x",
                               [&x](unsigned int slot, ULong64_t entry) {
                                  (void)slot;
                                  return x[entry];
                               })
                .DefineSlotEntry("y", [&y](unsigned int slot, ULong64_t entry) {
                   (void)slot;
                   return y[entry];
                });

   auto graph = dd.Graph("x", "y");

   // This tutorial is ran with multithreading enabled. The order in which points are inserted is not known, so to have a meaningful representation points are sorted.
   graph->Sort();
   graph->DrawClone("APL");
}
