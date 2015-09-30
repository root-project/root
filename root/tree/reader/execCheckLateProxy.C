{
TFile f("testVectorBranches.root");
auto t = (TTree*)f.Get("TestVectorBranches");
TTreeReader r(t);
r.SetEntry(3);
TTreeReaderArray<Float_t> vx(r, "vx");
r.Next();
cout << vx[1] << endl;
}
