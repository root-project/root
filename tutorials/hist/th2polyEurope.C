//This tutorial illustrates how to create an histogram with polygonal
//bins (TH2Poly), fill it and draw it. The initial data are stored
//in TMultiGraphs. They represent the european countries.
//The histogram filling is done according to a Mercator projection,
//therefore the bin contains should be proportional to the real surface
//of the countries.
//
//The initial data have been downloaded from: http://www.maproom.psu.edu/dcw/
//This database was developed in 1991/1992 and national boundaries reflect
//political reality as of that time.
//
//Author: Olivier Couet

void th2polyEurope()
{
   Int_t i,j;
   Double_t lon1 = -25;
   Double_t lon2 =  35;
   Double_t lat1 =  34;
   Double_t lat2 =  72;
   Double_t R = (lat2-lat1)/(lon2-lon1);
   Int_t W    = 800;
   Int_t H    = (Int_t)(R*800);
   gStyle->SetTitleX(0.2);
   gStyle->SetStatY(0.89);
   gStyle->SetStatW(0.15);

   // Canvas used to draw TH2Poly (the map)
   TCanvas *ce = new TCanvas("ce", "ce",0,0,W,H);
   ce->ToggleEventStatus();
   ce->SetGridx();
   ce->SetGridy();

   // Real surfaces taken from Wikipedia.
   const Int_t nx = 36;
   char *countries[nx] = { "france", "spain", "sweden", "germany", "finland",
                           "norway", "poland", "italy", "yugoslavia",
                           "united_kingdom", "romania", "belarus", "greece",
                           "czechoslovakia", "bulgaria", "iceland", "hungary",
                           "portugal", "austria", "ireland", "lithuania",
                           "latvia", "estonia", "denmark", "netherlands",
                           "switzerland", "moldova", "belgium", "albania",
                           "cyprus", "luxembourg", "andorra", "malta",
                           "liechtenstein", "san_marino", "monaco" };
   Float_t surfaces[nx] = { 547030, 505580, 449964, 357021, 338145, 324220,
                            312685, 301230, 255438, 244820, 237500, 207600,
                            131940, 127711, 110910, 103000, 93030, 89242, 83870,
                            70280, 65200, 64589, 45226, 43094, 41526, 41290,
                            33843, 30528, 28748, 9250, 2586, 468, 316, 160, 61,
                            2};

   TH1F *h = new TH1F("h","Countries' surfaces (in km^{2})",3,0,3);
   for (i=0; i<nx; i++) h->Fill(countries[i], surfaces[i]);
   h->LabelsDeflate();

   TFile *f;
   f = TFile::Open("http://root.cern.ch/files/europe.root");

   TH2Poly *p = new TH2Poly(
             "Europe",
             "Europe (bins' contains are normalize to the surfaces in km^{2})",
             lon1,lon2,lat1,lat2);
   p->GetXaxis()->SetNdivisions(520);

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

   gBenchmark->Start("Partitioning");
   p->ChangePartition(100, 100);
   gBenchmark->Show("Partitioning");

   // Fill TH2Poly according to a Mercator projection.
   gBenchmark->Start("Filling");
   for (i=0; i<500000; i++) {
      longitude = r.Uniform(dr*lon1,dr*lon2);
      latitude  = r.Uniform(-dr*90,dr*90);
      x         = rd*longitude;
      y         = 39*TMath::Log(TMath::Tan((TMath::Pi()/4)+(latitude/2)));
      p->Fill(x,y);
   }
   gBenchmark->Show("Filling");

   Int_t nbins = p->GetNumberOfBins();
   Double_t maximum = p->GetMaximum();
   printf("Nbins = %d Minimum = %g Maximum = %g Integral = %f \n",
           nbins,
           p->GetMinimum(),
           maximum,
           p->Integral());


   // h2 contains the surfaces computed from TH2Poly.
   TH1F *h2 = h->Clone();
   h2->Reset();
   for (j=0; j<nx; j++) {
      for (i=0; i<nbins; i++) {
         if (strstr(countries[j],p->GetBinName(i+1))) {
            h2->Fill(countries[j],p->GetBinContent(i+1));
            h2->SetBinError(j, p->GetBinError(i+1));
         }
      }
   }

   // Normalize the TH2Poly bin contents to the real surfaces.
   Double_t scale = surfaces[0]/maximum;
   for (i=0; i<nbins; i++) p->SetBinContent(i+1, scale*p->GetBinContent(i+1));

   gStyle->SetOptStat(1111);
   gStyle->SetPalette(1);
   p->Draw("COL");

   TCanvas *c1 = new TCanvas("c1", "c1",W+10,0,W-20,H);
   c1->SetRightMargin(0.047);

   Double_t scale = h->GetMaximum()/h2->GetMaximum();

   h->SetStats(0);
   h->Draw("L");
   h->SetLineColor(kRed-3);
   h->SetLineWidth(2);
   h->GetXaxis()->SetLabelFont(42);
   h->GetXaxis()->SetLabelSize(0.03);
   h->GetYaxis()->SetLabelFont(42);

   h2->Scale(scale);
   h2->Draw("E SAME");
   h2->SetMarkerStyle(20);
   h2->SetMarkerSize(0.8);

   TLegend *leg = new TLegend(0.5,0.67,0.92,0.8,NULL,"NDC");
   leg->SetTextFont(42);
   leg->SetTextSize(0.025);
   leg->AddEntry(h,"Real countries' surfaces from Wikipedia (in km^{2})","l");
   leg->AddEntry(h2,"Countries' surfaces from TH2Poly (with errors)","lp");
   leg->Draw();
}
