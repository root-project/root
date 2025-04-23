int assertRedirect() {
  gROOT->ProcessLine(".> Redirected.log");
  gROOT->ProcessLine(".x withRedirectedOutput.C");
  gROOT->ProcessLine(".>");
  ifstream inlog("Redirected.log");
  string logline;
  getline(inlog, logline);
  if (logline != "expected")
    return 1; // FAILURE!
  return 0;
}
