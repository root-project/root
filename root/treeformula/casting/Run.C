int Run(bool debug = false) {
  gSystem->Load("libTreePlayer"); 
  gROOT->ProcessLine(".L Simple.cxx+");
  gROOT->ProcessLine(".L Create.C");
  gROOT->ProcessLine(".L Read.C");
  
  Create(debug);
  return Read(debug);

}
