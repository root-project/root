{
  TFile::Open("stringarray.old.root");
  gROOT->ProcessLine(".x writefile.cxx+");
}
