int SimpleRun(bool debug = false) {
  gSystem->Load("libTreePlayer"); 
  gROOT->ProcessLine(".L Simple.cxx+");
  gROOT->ProcessLine(".L Create.C");
  gROOT->ProcessLine(".L Read.C");

#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(TString::Format("Create(%d);",debug));
   return gROOT->ProcessLine(TString::Format("Read(%d);",debug));
#else
  Create(debug);
   return Read(debug);
#endif
}

int Run(bool debug = false) {
   gSystem->Exit(!SimpleRun(debug));
   return 0;
}
