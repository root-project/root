class Vector3
{
   Double_t fX;
   Double_t fY;
   Double_t fZ;

public:
   Vector3() : fX(0),fY(0),fZ(0) {}

   Double_t x() { return fX; }
   Double_t y() { return fY; }
   Double_t z() { return fZ; }

   void SetXYZ(Double_t x, Double_t y, Double_t z) {
      fX = x;
      fY = y;
      fZ = z;
   }
};

void tv3Write() {
   //creates the Tree
   Vector3 *v = new Vector3();
   TFile *f = new TFile("v3.root","recreate");
   TTree *T = new TTree("T","v3 Tree");
   T->Branch("v3",&v,32000,1);
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
   Vector3 *v = 0;
   TFile *f = new TFile("v3.root");
   TTree *T = (TTree*)f->Get("T");
   T->SetBranchAddress("v3",&v);
   TH1F *h1 = new TH1F("x","x component of Vector3",100,-3,3);
   Long64_t nentries = T->GetEntries();
   for (Long64_t i=0;i<nentries;i++) {
      T->GetEntry(i);
      h1->Fill(v->x());
   }
   h1->Draw();
}

 void tv3Read2() {
   //second read example illustrating how to read one branch only
   Vector3 *v = 0;
   TFile *f = new TFile("v3.root");
   TTree *T = (TTree*)f->Get("T");
   T->SetBranchAddress("v3",&v);
   TBranch *by = T->GetBranch("fY");
   TH1F *h2 = new TH1F("y","y component of Vector3",100,-5,20);
   Long64_t nentries = T->GetEntries();
   for (Long64_t i=0;i<nentries;i++) {
      by->GetEntry(i);
      h2->Fill(v->y());
   }
   h2->Draw();
}

void tv3() {
  TCanvas *c1 = new TCanvas("c1","demo of Trees",10,10,600,800);
  c1->Divide(1,2);
  tv3Write();
  c1->cd(1);
  tv3Read1();
  c1->cd(2);
  tv3Read2();
}
