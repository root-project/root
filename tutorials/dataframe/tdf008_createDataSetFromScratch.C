/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -nodraw
/// This tutorial shows how to create a dataset from scratch with TDataFrame
/// \macro_code
///
/// \date June 2017
/// \author Danilo Piparo

void tdf008_createDataSetFromScratch()
{
   // We create an empty data frame of 100 entries
   ROOT::Experimental::TDataFrame tdf(100);

   // We now fill it with random numbers
   gRandom->SetSeed(1);
   auto tdf_1 = tdf.Define("rnd", []() { return gRandom->Gaus(); });

   // And we write out the dataset on disk
   tdf_1.Snapshot("randomNumbers", "tdf008_createDataSetFromScratch.root");
}
