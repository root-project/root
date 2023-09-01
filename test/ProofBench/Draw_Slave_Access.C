void Draw_Slave_Access(TTree *perfstats);

void Draw_Slave_Access(const Char_t* filename,
                       const Char_t* perfstatsname = "PROOF_PerfStats") {
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
      Draw_Slave_Access(perfstats);
   } else {
      cout << "No Tree named " << perfstatsname
           << " found in file " << filename << endl;
   }

   delete perfstats;
   f.Close();
}

void Draw_Slave_Access(TTree *perfstats) {
   // Draws a graph of the number of slaves
   // accessing files as a function of time
   // from a tree made from TPerfStats

   gROOT->SetStyle("Plain");
   gStyle->SetNdivisions(505, "X");
   gStyle->SetNdivisions(505, "Y");
   gStyle->SetTitleFontSize(0.07);

   if(!TString(gSystem->GetLibraries()).Contains("Proof"))
      gSystem->Load("libProof.so");

   if (!perfstats) {
      cout << "Input tree invalid" << endl;
      return;
   }

   //make sure PerfEvents branch exists
   if (!perfstats->FindBranch("PerfEvents")) {
      cout << "Input tree does not have a PerfEvents branch" << endl;
      return;
   }

   Int_t nentries = perfstats->GetEntries();
   TPerfEvent pe;
   TPerfEvent* pep = &pe;
   perfstats->SetBranchAddress("PerfEvents",&pep);

   //make graph
   TGraph* graph = new TGraph(1);
   graph->SetName("all");
   graph->SetTitle("Global File Access");
   //Set first point to 0
   graph->SetPoint(0,0,0);

   for(Int_t entry=0;entry<nentries;entry++){
      perfstats->GetEntry(entry);

      //Check if it is a file event
      if(pe.fType==TVirtualPerfStats::kFile){

         Double_t time = (pe.fTimeStamp.GetSec())+
                         (pe.fTimeStamp.GetNanoSec())*1e-9;
         Int_t npoints=graph->GetN();
         Double_t y = (graph->GetY())[npoints-1];
         graph->SetPoint(npoints,time,y);
         if(pe.fIsStart==kTRUE) y++;
         else y--;
         graph->SetPoint(npoints+1,time,y);

      }
   }

   //reset branch address to 0 since address will be invalid after leaving macro
   perfstats->SetBranchAddress("PerfEvents", 0);

   // Draw Canvas
   TCanvas* canvas = new TCanvas("slave_access",
                                 "Number of Slaves Accessing Nodes vs Time");
   canvas->cd();
   graph->GetXaxis()->SetTitle("Time [s]");
   graph->GetYaxis()->SetTitle("Number of Slaves Accessing Files");
   graph->Draw("AL");
   gPad->Update();

   // center title
   TPaveText* titlepave =
      dynamic_cast<TPaveText*>(gPad->GetListOfPrimitives()->FindObject("title"));
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
