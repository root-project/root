/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
/// This tutorial shows how to create a dataset from scratch with RDataFrame
///
/// \macro_code
///
/// \date June 2017
/// \author Danilo Piparo

void df008_createDataSetFromScratch()
{
   // We create an empty data frame of 100 entries
   ROOT::RDataFrame tdf(100);

   // We now fill it with random numbers
   gRandom->SetSeed(1);
   auto tdf_1 = tdf.Define("rnd", []() { return gRandom->Gaus(); });

   // And we write out the dataset on disk
   tdf_1.Snapshot("randomNumbers", "df008_createDataSetFromScratch.root");
}
