//This tutorial illustrates how to create an histogram with polygonal
//bins (TH2Poly), fill it and draw it. The initial data are stored
//in TMultiGraphs. They represent the USA.
//The initial data have been downloaded from: http://www.maproom.psu.edu/dcw/
//Author: Olivier Couet

void th2polyUSA()
{
   TCanvas *usa = new TCanvas("USA", "USA");
   usa->ToggleEventStatus();

   Double_t x1 = -130;
   Double_t x2 = -65;
   Double_t y1 = 24;
   Double_t y2 = 50;
   TH2Poly *p = new TH2Poly("USA","USA",x1,x2,y1,y2);

   TFile *f;
   f = TFile::Open("http://root.cern.ch/files/usa.root");

   TMultiGraph *mg;
   TKey *key;
   TIter nextkey(gDirectory->GetListOfKeys());
   while (key = (TKey*)nextkey()) {
      obj = key->ReadObj();
      if (obj->InheritsFrom("TMultiGraph")) {
         mg = (TMultiGraph*)obj;
         p->AddBin(mg);
      }
   }

   TRandom r;
   Double_t px,py;

   gBenchmark->Start("Partitioning");
   p->ChangePartition(100, 100);
   gBenchmark->Show("Partitioning");

   for (Int_t i=0; i<500000; i++) {
      px = (x2-x1)*r.Gaus(2.,1)+(x2+x1)/2;
      py = (y2-y1)*r.Gaus(2.,1)+(y2+y1)/2;
      p->Fill(px,py);
   }
   gBenchmark->Show("Filling");

   gStyle->SetOptStat(11);
   gStyle->SetPalette(1); 
   p->Draw("COL");
   printf("Nbins = %d Minimum = %g Maximum = %g Integral = %f \n",
          p->GetNumberOfBins(),
          p->GetMinimum(),
          p->GetMaximum(),
          p->Integral("width"));
}
