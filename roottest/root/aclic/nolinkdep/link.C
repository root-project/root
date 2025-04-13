void link(int what) {
   switch (what) {
      case 1:
         gROOT->ProcessLine(".L single.C+");
         gROOT->ProcessLine(".L script1.C+");
         gROOT->ProcessLine(".L script2.C+");
         break;
      case 10:
         gROOT->ProcessLine(".L script.C+");
         gSystem->CopyFile("script2.C","script.C");
         gROOT->ProcessLine(".L script.C+");         
         break;
      default:
         break;
   }
}