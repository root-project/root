void Run(bool skipKnownFail) {
  gROOT->ProcessLine(".L sync.C");
  gSystem->Exit(!sync(skipKnownFail));
}
