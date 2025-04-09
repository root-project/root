
void write_int () {
   TFile *file = TFile::Open("mix_int.root","RECREATE");
   TTree *tree = new TTree("mix","");
   Int_t myint = 'c';
   tree->Branch("values",&myint);
   tree->Fill();
   myint ='d';
   tree->Fill();
   file->Write();
   delete file;
}

void write_char () {
   TFile *file = TFile::Open("mix_char.root","RECREATE");
   TTree *tree = new TTree("mix","");
   Char_t mychar = 'a';
   tree->Branch("values",&mychar);
   tree->Fill();
   mychar ='b';
   tree->Fill();
   file->Write();
   delete file;
}

void scan_chain() {
   TChain c("mix");
   c.AddFile("mix_char.root");
   c.AddFile("mix_int.root");
   c.Scan();
}

void merge() {
   TChain c("mix");
   c.AddFile("mix_char.root");
   c.AddFile("mix_int.root");
   TFile *m = TFile::Open("mix_merge.root","RECREATE");
   c.CloneTree(-1,"fast");
   m->Write();
   delete m;
}

void scan_merge() {
   TChain c("mix");
   c.AddFile("mix_merge.root");
   c.Scan();
}

void write() {
   write_char();
   write_int();
   scan_chain();
}

void runbadmix() {
   write();
   merge();
   scan_chain();
}
