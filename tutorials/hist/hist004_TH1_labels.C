/// \file
/// \ingroup tutorial_hist
/// \notebook
/// 1D histograms with alphanumeric labels.
///
/// A TH1 can have named bins that are filled with the method overload TH1::Fill(const char*, double)
///
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author Rene Brun

TCanvas *hist004_TH1_labels()
{
   // Create the histogram
   const std::array people{"Jean",    "Pierre", "Marie",    "Odile",   "Sebastien", "Fons",  "Rene",
                           "Nicolas", "Xavier", "Greg",     "Bjarne",  "Anton",     "Otto",  "Eddy",
                           "Peter",   "Pasha",  "Philippe", "Suzanne", "Jeff",      "Valery"};
   // Start with an arbitrary amount of bins and an arbitrary range, but this will be extended thanks to SetCanExtend().
   int nBins = 3;
   double rangeMin = 0.0;
   double rangeMax = 3.0;
   auto *h = new TH1D("h", "test", nBins, rangeMin, rangeMax);
   // Disable the default stats box when drawing this histogram
   h->SetStats(0);
   h->SetFillColor(38);
   // Allow both axes to extend past the initial range we gave in the constructor
   h->SetCanExtend(TH1::kAllAxes);
   // Fill the Y axis with arbitrary values, a random amount per bin
   TRandom3 rng;
   for (int i = 0; i < 5000; i++) {
      int r = rng.Rndm() * 20;
      // `Fill()` called with a const char* as the first argument will add a value to the bin with that name,
      // creating it if it doesn't exist yet.
      h->Fill(people[r], 1);
   }
   // Remove empty bins
   h->LabelsDeflate();

   auto *c1 = new TCanvas("c1", "demo bin labels", 10, 10, 900, 500);
   // Enable the grid in the plot
   c1->SetGrid();
   c1->SetTopMargin(0.15);

   // Draw the histogram
   h->Draw();

   // Draw a boxed text
   // "brNDC" = coordinates draw bottom-right shadow and pass the coordinates in normalized device coordinates
   auto *pt = new TPaveText(0.7, 0.85, 0.98, 0.98, "brNDC");
   pt->SetFillColor(18);
   pt->SetTextAlign(12);
   pt->AddText("Use the axis Context Menu LabelsOption");
   pt->AddText(" \"a\"   to sort by alphabetic order");
   pt->AddText(" \">\"   to sort by decreasing values");
   pt->AddText(" \"<\"   to sort by increasing values");
   pt->Draw();

   return c1;
}
