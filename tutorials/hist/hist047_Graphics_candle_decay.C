/// \file
/// \ingroup tutorial_hist
/// \notebook
/// \preview Candle Decay, illustrate a time development of a certain value.
///
/// \macro_image (tcanvas_js)
/// \macro_code
///
/// \author Georg Troska

void hist047_Graphics_candle_decay()
{
   // create a new 2D histogram with x-axis title probability density and y-axis title time
   // it is of a type TH2I (2D histogram with integer values)
   auto *hist = new TH2I("hist", "Decay; probability density; time", 1000, 0, 1000, 20, 0, 20);
   TRandom rand; // create a random number generator outside the loop to avoid reseeding

   for (int iBin = 0; iBin < 19; iBin++) {
      for (int j = 0; j < 1000000; j++) {
         // generate a random number from a 2D Gaussian distribution
         // a simulation of a time development of a certain value
         float myRand = rand.Gaus(350 + iBin * 8, 20 + 2 * iBin);
         hist->Fill(myRand, iBin);
      }
   }
   hist->SetBarWidth(3);
   hist->SetFillStyle(0);
   hist->SetFillColor(kGray);
   hist->SetLineColor(kBlue);
   //  create a new canvas
   auto *can = new TCanvas("can", "Candle Decay", 800, 600);
   can->Divide(2, 1); // divide the canvas into 2 pads (2 rows, 1 column)

   can->cd(1);
   hist->Draw("violiny(112000000)");
   can->cd(2);
   auto *hist2 = static_cast<TH2I *>(hist->Clone("hist2"));
   hist2->SetBarWidth(0.8);
   // There are six predefined candle-plot representations: (X can be replaced by Y)
   //  "CANDLEX1": Standard candle (whiskers cover the whole distribution)
   //  "CANDLEX2": Standard candle with better whisker definition + outliers. It is a good compromise
   //  "CANDLEX3": Like candle2 but with a mean as a circle. It is easier to distinguish mean and median
   //  "CANDLEX4": Like candle3 but showing the uncertainty of the median as well (notched candle plots). For bigger
   //  datasets per candle "CANDLEX5": Like candle2 but showing all data points. For very small datasets "CANDLEX6":
   //  Like candle2 but showing all datapoints scattered. For huge datasets
   hist2->DrawCopy("candley2");
}
