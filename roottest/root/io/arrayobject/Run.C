void Run(char* file= "Event.3.2.0.root") {
gROOT->ProcessLine(".L libFoo.so");

  TFile *h = new TFile(file, "READ");
  TTree * tree = (TTree*)h->Get("T");
  bar *b = 0;
  tree->SetBranchAddress("a/b",&b);
  
  tree->GetEntry(3);
  b->print();
  
}
