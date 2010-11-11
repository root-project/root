//This tutorial illustrates how to create an histogram with polygonal
//bins (TH2Poly), fill it and draw it using GL. The initial data are stored
//in TMultiGraphs. They represent the USA.
//
//The initial data have been downloaded from: http://www.maproom.psu.edu/dcw/
//This database was developed in 1991/1992 and national boundaries reflect
//political reality as of that time.
//
//Author: Olivier Couet

void th2polyUSA()
{
	gStyle->SetCanvasPreferGL(true);
   TCanvas *usa = new TCanvas("USA", "USA");
   usa->ToggleEventStatus();
   Double_t lon1 = -130;
   Double_t lon2 = -65;
   Double_t lat1 = 24;
   Double_t lat2 = 50;
   TH2Poly *p = new TH2Poly("USA","USA",lon1,lon2,lat1,lat2);

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
   Double_t longitude, latitude;
   Double_t x, y, dr = TMath::Pi()/180, rd = 180/TMath::Pi();

   p->ChangePartition(100, 100);

   for (Int_t i=0; i<500000; i++) {
      longitude = r.Uniform(dr*lon1,dr*lon2);
      latitude  = r.Uniform(-dr*90,dr*90);
      x         = rd*longitude;
      y         = 39*TMath::Log(TMath::Tan((TMath::Pi()/4)+(latitude/2)));
      p->Fill(x,y);
   }

   gStyle->SetOptStat(11);
   gStyle->SetPalette(1); 
   p->Draw("glhp");
}
