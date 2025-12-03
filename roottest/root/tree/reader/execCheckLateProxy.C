int execCheckLateProxy(){
TString hSimplePath = "hsimple.root";
TFile f(hSimplePath);
auto t = (TTree*)f.Get("ntuple");
TTreeReader r(t);
r.SetEntry(3);
TTreeReaderValue<Float_t> px(r, "px");
r.Next();
cout << *px << endl;
return 0;
}

