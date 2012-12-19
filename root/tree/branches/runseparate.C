{
   int i = 3;
   float f = 7;

   TFile *file = TFile::Open("treeobj.root","RECREATE");
   TTree *t = new TTree("T","T");

   t->Branch("i",&i);
   TBranch *b = t->Branch("f",&f,100);
   b->SetFile("branch.root");
   for(int i=0; i<20; ++i) {
      t->Fill();
   }
   
   file->Write();
   TFile *other = (TFile*)gROOT->GetListOfFiles()->FindObject("branch.root");
   other->Save();
   
   delete other;
   delete file;
   
   file = TFile::Open("treeobj.root");
   file->GetObject("T",t);
   t->GetEntry(0);
   delete file;
   if (gROOT->GetListOfFiles()->GetEntries() > 0) {
      printf("Some files where not closed:\n");
      gROOT->GetListOfFiles()->ls();
   }
   
   file = TFile::Open(TString::Format("%s/treeobj.root",gSystem->pwd()));
   gSystem->cd("..");
   file->GetObject("T",t);
   t->GetEntry(0);
   delete file;
   if (gROOT->GetListOfFiles()->GetEntries() > 0) {
      printf("Some files where not closed:\n");
      gROOT->GetListOfFiles()->ls();
   }
}
