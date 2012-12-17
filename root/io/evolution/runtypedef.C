{
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
   gROOT->ProcessLine(".L typedefWrite.C+");
   TFile *f = new TFile("typedef.root");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(
                      "UHTTimeFitter *u;"
                      "f->GetObject(\"myobject\",&u);"
                      );
#else
   UHTTimeFitter *u;
   f->GetObject("myobject",&u);
#endif
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   }
#endif
}
