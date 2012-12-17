{
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif

#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L Event_3.cxx+");
#else
   #include "Event_3.cxx+"
#endif
   TFile *f = TFile::Open("Event_2.root");
   TTree *t; f->GetObject("T",t);
   t->GetEntry(0);
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   }
#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
