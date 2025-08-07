void link() {
   gROOT->ProcessLine(".L single.C+");
   gSystem->CopyFile("script2.C","script.C");
   gROOT->ProcessLine(".L script.C+");
}