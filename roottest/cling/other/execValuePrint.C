// Make sure printValue works.
{
  gROOT->ProcessLine("TObject b");
  gROOT->ProcessLine("TNamed n(\"a\", \"b\")");
  gROOT->ProcessLine("TH1F h");
  gROOT->ProcessLine("TDatime d(950130, 124559)");
  gROOT->ProcessLine("TString s(\"123ABC\")");
  gROOT->ProcessLine("TFitResult fr");
  gROOT->ProcessLine("TFitResultPtr frp");
  0;
}
