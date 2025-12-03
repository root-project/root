void Run(const char* filename = "Event.3.2.0.root")
{
  auto f = TFile::Open(filename);
  TTree * tree = (TTree*) f->Get("T");
  bar *b = nullptr;
  tree->SetBranchAddress("a/b",&b);

  tree->GetEntry(3);
  b->print();
}
