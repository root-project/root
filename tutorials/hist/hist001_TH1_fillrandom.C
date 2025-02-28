/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Fill a 1D histogram with random values using predefined functions.
///
/// \macro_code
///
/// \date November 2024
/// \author Giacomo Parolini

void hist001_TH1_fillrandom()
{
   // Create a one dimensional histogram and fill it with a gaussian distribution
   int nBins = 200;
   double rangeMin = 0.0;
   double rangeMax = 10.0;
   TH1D h1d("h1d", "Test random numbers", nBins, rangeMin, rangeMax);

   // "gaus" is a predefined ROOT function. Here we are filling the histogram with
   // 10000 values sampled from that distribution.
   h1d.FillRandom("gaus", 10000);

   // Open a ROOT file and save the histogram
   auto myfile = std::unique_ptr<TFile>(TFile::Open("fillrandom.root", "RECREATE"));
   myfile->WriteObject(&h1d, h1d.GetName());
}
