//This tutorial illustrates how to create an histogram with polygonal
//bins (TH2Poly), fill it and draw it. The initial data are stored
//in TMultiGraphs. They represent the european countries.
//The initial data have been downloaded from: http://www.maproom.psu.edu/dcw/
//Author: Olivier Couet

void th2polyEurope()
{
   TCanvas *ce = new TCanvas("ce", "ce");
   ce->ToggleEventStatus();

   TFile *f;
   f = TFile::Open("http://root.cern.ch/files/europe.root");

   TH2Poly *p = new TH2Poly("europe","europe",-30,40,30,75);

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

   gBenchmark->Start("Filling");
   for (Int_t i=0; i<500000; i++) {
      px = r.Gaus(2.,1)*30;
      py = 20*r.Gaus(2.,1)+20;
      p->Fill(px,py);
   }
   gBenchmark->Show("Filling");

   gStyle->SetOptStat(1111);
   gStyle->SetPalette(1);
   p->Draw("COL");
   printf("Nbins = %d Minimum = %g Maximum = %g Integral = %f \n",
           p->GetNumberOfBins(),
           p->GetMinimum(),
           p->GetMaximum(),
           p->Integral("width"));
}
