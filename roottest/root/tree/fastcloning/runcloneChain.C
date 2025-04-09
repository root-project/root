{
#ifdef ClingWorkAroundIncorrectTearDownOrder
if (1) {
#endif
  TFile OriginalFile("ver_40200.root");
  TFile CloneFile("CloneTree.root");

  TTree* OriginalTree = (TTree*) OriginalFile.Get("NtpSt");
  TTree* CloneTree     = (TTree*) CloneFile.Get("NtpSt");

  cout << "Number of entries in original " << OriginalTree->GetEntries() << endl;
  cout << "Number of entries in double copy " << CloneTree->GetEntries() << endl;

  if (OriginalTree->GetEntries()*2==CloneTree->GetEntries()) {
#ifdef ClingWorkAroundBrokenUnnamedReturn
     gApplication->Terminate(0);
#else
     return 0;
#endif
  } else {
     cout << "The 2nd number should have been twice the first\n";
#ifdef ClingWorkAroundBrokenUnnamedReturn
     gApplication->Terminate(1);
#else
     return 1;
#endif
  }
#ifdef ClingWorkAroundIncorrectTearDownOrder
}
#endif
}
