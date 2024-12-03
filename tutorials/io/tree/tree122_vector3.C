/// \file
/// \ingroup tutorial_tree
/// \notebook -nodraw
/// Write and read a Vector3 class in a tree.
///
/// \macro_code
///
/// \author The ROOT Team

class Vector3
{
   Double_t fX;
   Double_t fY;
   Double_t fZ;

public:
   Vector3() : fX(0), fY(0), fZ(0) {}

   Double_t x() { return fX; }
   Double_t y() { return fY; }
   Double_t z() { return fZ; }

   void SetXYZ(Double_t x, Double_t y, Double_t z) {
      fX = x;
      fY = y;
      fZ = z;
   }
};

void write_vector3()
{
   //creates the Tree
   auto v = new Vector3();
   auto f = TFile::Open("vector3.root", "recreate");
   auto T = new TTree("T", "vector3 Tree");
   T->Branch("v3", &v, 32000, 1);
   TRandom r;
   for (Int_t i=0; i<10000; i++) {
      v->SetXYZ(r.Gaus(0, 1), r.Landau(0, 1), r.Gaus(100, 10));
      T->Fill();
   }
   T->Write();
   T->Print();
   delete f;
}

void read_all_vector3()
{
   //first read example showing how to read all branches
   Vector3 *v = 0;
   auto f = TFile::Open("vector3.root");
   auto T = f->Get<TTree>("T");
   T->SetBranchAddress("v3", &v);
   auto h1 = new TH1F("x", "x component of Vector3", 100, -3, 3);
   Long64_t nentries = T->GetEntries();
   for (Long64_t i=0; i<nentries; i++) {
      T->GetEntry(i);
      h1->Fill(v->x());
   }
   h1->Draw();
}

 void read_branch_vector3()
 {
   //second read example illustrating how to read one branch only
   Vector3 *v = 0;
   auto f = TFile::Open("vector3.root");
   auto T = f->Get<TTree>("T");
   T->SetBranchAddress("v3", &v);
   auto by = T->GetBranch("fY");
   auto h2 = new TH1F("y", "y component of Vector3", 100, -5, 20);
   Long64_t nentries = T->GetEntries();
   for (Long64_t i=0; i<nentries; i++) {
      by->GetEntry(i);
      h2->Fill(v->y());
   }
   h2->Draw();
}

void tree122_vector3()
{
   auto c1 = new TCanvas("c1", "demo of Trees", 10, 10, 600, 800);
   c1->Divide(1, 2);
   write_vector3();
   c1->cd(1);
   read_all_vector3();
   c1->cd(2);
   read_branch_vector3();
}
