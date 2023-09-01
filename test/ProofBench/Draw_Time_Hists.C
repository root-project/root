void Draw_Time_Hists(TTree* t);

void Draw_Time_Hists(const Char_t* filename, const Char_t* perfstatsname) {
   // Open the file called filename and get
   // the perftstats tree (name perfstatsname)
   // from the file

   if(!TString(gSystem->GetLibraries()).Contains("Proof"))
      gSystem->Load("libProof.so");

   TFile f(filename);
   if (f.IsZombie()) {
      cout << "Could not open file " << filename << endl;
      return;
   }

   TTree* perfstats = dynamic_cast<TTree*>(f.Get(perfstatsname));
   if (perfstats) {
      Draw_Time_Hists(perfstats);
   } else {
      cout << "No Tree named " << perfstatsname
         << " found in file " << filename << endl;
   }

   delete perfstats;
   f.Close();
}

void Draw_Time_Hists(TTree* t) {
   // Draw processing time, CPU time, and latency per packet
   // distributions from the input tree

   if (!t) {
      cout << "Invalid input tree" << endl;
      return;
   }

   gROOT->SetStyle("Plain");
   gStyle->SetOptStat(0);
   gStyle->SetNdivisions(505);
   gStyle->SetTitleFontSize(0.1);

   if(!TString(gSystem->GetLibraries()).Contains("Proof"))
      gSystem->Load("libProof.so");

   TCanvas* canvas = new TCanvas("ProfHists","Profile Histograms",800,600);
   canvas->Divide(3,1);

   canvas->cd(1);
   gPad->SetLogy();
   t->Draw("fProcTime");
   TH1* h = dynamic_cast<TH1*>(gROOT->FindObject("htemp"));
   h->SetTitle("Processing Time per Packet");
   h->GetXaxis()->SetTitle("Processing Time [s]");

   gPad->Update();
   TPaveText* titlepave = 0;
   titlepave = dynamic_cast<TPaveText*>(gPad->GetListOfPrimitives()->FindObject("title"));
   if (titlepave) {
      Double_t x1ndc = titlepave->GetX1NDC();
      Double_t x2ndc = titlepave->GetX2NDC();
      titlepave->SetX1NDC((1.0-x2ndc+x1ndc)/2.);
      titlepave->SetX2NDC((1.0+x2ndc-x1ndc)/2.);
      titlepave->SetBorderSize(0);
      gPad->Update();
   }
   gPad->Modified();

   canvas->cd(2);
   gPad->SetLogy();
   t->Draw("fCpuTime");
   h = dynamic_cast<TH1*>(gROOT->FindObject("htemp"));
   h->SetTitle("CPU Time per Packet");
   h->GetXaxis()->SetTitle("CPU Time [s]");

   gPad->Update();
   titlepave = dynamic_cast<TPaveText*>(gPad->GetListOfPrimitives()->FindObject("title"));
   if (titlepave) {
      Double_t x1ndc = titlepave->GetX1NDC();
      Double_t x2ndc = titlepave->GetX2NDC();
      titlepave->SetX1NDC((1.0-x2ndc+x1ndc)/2.);
      titlepave->SetX2NDC((1.0+x2ndc-x1ndc)/2.);
      titlepave->SetBorderSize(0);
      gPad->Update();
   }
   gPad->Modified();

   canvas->cd(3);
   gPad->SetLogy();
   t->Draw("fLatency");
   h = dynamic_cast<TH1*>(gROOT->FindObject("htemp"));
   h->SetTitle("Request Packet Latency");
   h->GetXaxis()->SetTitle("Latency [s]");

   gPad->Update();
   titlepave = dynamic_cast<TPaveText*>(gPad->GetListOfPrimitives()->FindObject("title"));
   if (titlepave) {
      Double_t x1ndc = titlepave->GetX1NDC();
      Double_t x2ndc = titlepave->GetX2NDC();
      titlepave->SetX1NDC((1.0-x2ndc+x1ndc)/2.);
      titlepave->SetX2NDC((1.0+x2ndc-x1ndc)/2.);
      titlepave->SetBorderSize(0);
      gPad->Update();
   }
   gPad->Modified();

}
