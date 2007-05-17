{
   gROOT->ProcessLine(".L sel01.C");
   sel01 isel;
   TFile *f = TFile::Open("Event1.root");
   TTree *tree; f->GetObject("T1",tree);
   
   tree->Process(&isel);
   
   gROOT->ProcessLine(".L sel01.C+");
   sel01 csel;
   
   TFile *f = TFile::Open("Event1.root");
   TTree *tree; f->GetObject("T1",tree);
   
   tree->Process(&csel);
  
}
   