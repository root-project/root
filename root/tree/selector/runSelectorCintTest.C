{
   // Avoid loading the library
   gInterpreter->UnloadLibraryMap("sel01_C");

   gROOT->ProcessLine(".L sel01.C");
   sel01 isel;
   TFile *f = TFile::Open("Event1.root");
   TTree *tree; f->GetObject("T1",tree);
   
   tree->Process(&isel);

   c = new TChain("T1");
   c->Add("Event1.root");
   c->Process(&isel);
   
   gROOT->ProcessLine(".L sel01.C+");
   sel01 csel;
   
   TFile *f = TFile::Open("Event1.root");
   TTree *tree; f->GetObject("T1",tree);
   
   tree->Process(&csel);

   c = new TChain("T1");
   c->Add("Event1.root");
   c->Process(&csel);

}
   
