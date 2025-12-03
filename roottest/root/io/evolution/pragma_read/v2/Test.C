void
Test(bool useTree=true, bool useObj=true)
{
  gSystem->Load("libRoottestIoEvolutionPragmaV2");


  TFile* file = TFile::Open("test.root", "READ");
  if (!file) {
    Error("Test", "Failed to open file");
    return;
  }

  std::cout << "=== Reading new style object ===" << std::endl;
  AliAODForwardMult* mt = 0;
  AliAODForwardMult* ms = 0;
  if (useTree) {
    TTree* tree = static_cast<TTree*>(file->Get("T"));
    tree->SetBranchAddress("Forward.", &mt);
    tree->GetEntry(0);
  }
  if (useObj)
    ms = static_cast<AliAODForwardMult*>(file->Get("Forward"));

  std::cout << "=== Read from file " << (useTree ? "(from TTree) " : "")
	    << "===" << std::endl;
  if (useTree) mt->Print();
  if (useObj) ms->Print();

  file->Close();
  std::cout << "=== End of Test ===" << std::endl;
}

