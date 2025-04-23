{
   gROOT->ProcessLine(".L foo.C+");
   gROOT->ProcessLine(".L bar.C+");
   gROOT->ProcessLine(".L main.C+");
   run();
}
