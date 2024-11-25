/// \file
/// \ingroup tutorial_hist
/// Hello World example for TH1
///
/// Shows how to create, fill and write a histogram to a ROOT file.
///
/// \macro_code
/// \macro_output
///
/// \date November 2024
/// \author Giacomo Parolini (CERN)

void hist000_TH1_first()
{
   // Open the file to write the histogram to
   auto outFile = std::unique_ptr<TFile>(TFile::Open("outfile.root", "RECREATE"));

   // Create the histogram object
   // There are several constructors you can use (\see TH1). In this example we use the
   // simplest one, accepting a number of bins and a range.
   int nBins = 30;
   double rangeMin = 0.0;
   double rangeMax = 10.0;
   TH1D histogram("histogram", "My first ROOT histogram", nBins, rangeMin, rangeMax);

   // Fill the histogram. In this simple example we use a fake set of data.
   // The 'D' in TH1D stands for 'double', so we fill the histogram with doubles.
   // In general you should prefer TH1D over TH1F unless you have a very specific reason
   // to do otherwise.
   const std::array values{1, 2, 3, 3, 3, 4, 3, 2, 1, 0};
   for (double val : values) {
      histogram.Fill(val);
   }

   // Write the histogram to `outFile`.
   outFile->WriteObject(&histogram, histogram.GetName());

   // When the TFile goes out of scope it will close itself and write its contents to disk.
}
