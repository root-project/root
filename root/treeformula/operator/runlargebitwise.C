{
  unsigned int ui = 0x80000001;
  cout << (ui & 0x7fffffff) << endl;

  TTree* tree = new TTree("tree", "tree");
  tree->Branch("ui", &ui, "ui/i");
  tree->Fill();
  tree->Scan("ui:ui & 0x7fffffff", "", "col=16lx:16lx");
#ifdef ClingWorkAroundBrokenUnnamedReturn
  gApplication->Terminate(0);
#else
  return 0;
#endif
}
