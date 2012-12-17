{
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
   gROOT->ProcessLine(".L typedefWrite.C+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(
      "TFile *f = new TFile(\"typedef.root\");"
      "UHTTimeFitter *u;"
      "f->GetObject(\"myobject\",u);"
      );
#else
   TFile *f = new TFile("typedef.root");
   UHTTimeFitter *u;
   f->GetObject("myobject",u);
#endif
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   }
#endif
}
