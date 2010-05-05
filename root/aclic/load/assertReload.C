{
  gErrorIgnoreLevel = kError;
  gROOT->ProcessLine(".x ReloadScript.C++(\"Reload.root\")");
  gROOT->ProcessLine(".x ReloadScript.C++(\"Reload.root\")");
  return 0;
}
