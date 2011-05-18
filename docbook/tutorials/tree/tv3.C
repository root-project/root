void tv3() {
   gSystem->Load("libPhysics");
   TCanvas *c1 = new TCanvas("c1","demo of Trees",10,10,600,800);
   c1->Divide(1,2);
   tv3Write();
   c1->cd(1);
   tv3Read1();
   c1->cd(2);
   tv3Read2();
}
void tv3Write() {
   //creates the Tree
   TVector3 *v = new TVector3();
   TVector3::Class()->IgnoreTObjectStreamer();
   TFile *f = new TFile("v3.root","recreate");
   TTree *T = new TTree("T","v3 Tree");
   T->Branch("v3","TVector3",&v,32000,1);
   TRandom r;
   for (Int_t i=0;i<10000;i++) {
      v->SetXYZ(r.Gaus(0,1),r.Landau(0,1),r.Gaus(100,10));
      T->Fill();
   }
   T->Write();
   T->Print();
   delete f;
}
void tv3Read1() {
   //first read example showing how to read all branches
   TVector3 *v = 0;
   TFile *f = new TFile("v3.root");
   TTree *T = (TTree*)f->Get("T");
   T->SetBranchAddress("v3",&v);
   TH1F *h1 = new TH1F("x","x component of TVector3",100,-3,3);
   Int_t nentries = Int_t(T->GetEntries());
   for (Int_t i=0;i<nentries;i++) {
      T->GetEntry(i);
      h1->Fill(v->x());
   }
   h1->Draw();
}

 void tv3Read2() {
   //second read example illustrating how to read one branch only
   TVector3 *v = 0;
   TFile *f = new TFile("v3.root");
   TTree *T = (TTree*)f->Get("T");
   T->SetBranchAddress("v3",&v);
   TBranch *by = T->GetBranch("fY");
   TH1F *h2 = new TH1F("y","y component of TVector3",100,-5,20);
   Int_t nentries = Int_t(T->GetEntries());
   for (Int_t i=0;i<nentries;i++) {
      by->GetEntry(i);
      h2->Fill(v->y());
   }
   h2->Draw();
}
  
