void
Test(bool useTree=true, bool useObj = true)
{
  gSystem->Load("libRoottestIoEvolutionPragmaV1");

  TFile* file = TFile::Open("test.root", "RECREATE");
  if (!file) {
    Error("Test", "Failed to open file");
    return;
  }


  std::cout << "=== Creating old-style object ===" << std::endl;
  AliAODForwardMult* m    = new AliAODForwardMult(false);
  m->SetTriggerMask(AliAODForwardMult::kNSD);
  m->SetIpZ(-5);
  m->SetSNN(2760);
  m->SetSystem(2);
  m->SetCentrality(30);
  m->Print();

  if (useTree) {
    TTree* tree = new TTree("T", "T");
    tree->Bronch("Forward.", "AliAODForwardMult", &m, 32000, 99);
    tree->Fill();
  }
  if (useObj) {
    m->Write();
  }

  std::cout << "=== Wrote to file " << (useTree ? "(in TTree) " : "")
	    << "===" << std::endl;
  file->Write();
  file->Close();
  std::cout << "=== End of Test ===" << std::endl;
}

