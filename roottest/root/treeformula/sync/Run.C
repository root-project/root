void Run(bool skipKnownFail) {
  gROOT->ProcessLine(".L sync.C");
#ifdef ClingWorkAroundMissingDynamicScope
  gROOT->ProcessLine(TString::Format("gSystem->Exit(!sync(%d));",skipKnownFail));
#else
  gSystem->Exit(!sync(skipKnownFail));
#endif
}
