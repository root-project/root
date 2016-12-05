/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Candle Decay, illustrate a time development of a certain value.
///
/// \macro_image
/// \macro_code
///
/// \author Georg Troska

void candledecay()
{
   TCanvas *c1 = new TCanvas("c1","Candle Decay",800,600);
   c1->Divide(2,1);
   TRandom *rand = new TRandom();
   TH2I *h1 = new TH2I("h1","Decay",1000,0,1000,20,0,20);

   float myRand;
   for (int i = 0; i < 17; i++) {
      for (int j = 0; j < 1000000; j++) {
         myRand = rand->Gaus(350+i*5,5+5*i);
         h1->Fill(myRand,i);
      }
   }
   h1->SetBarWidth(4);
   h1->SetFillStyle(0);
   h1->SetFillColor(kGray);
   h1->SetLineColor(kBlue);
   h1->GetYaxis()->SetTitle("time");
   h1->GetXaxis()->SetTitle("probability density");

   c1->cd(1);
   h1->Draw("violiny(12000000)");
   c1->cd(2);
   TH2I *h2 = (TH2I*)h1->Clone("h2");
   h2->SetBarWidth(0.8);
   h2->DrawCopy("candley2");
}
