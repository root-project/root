#ifdef ClingWorkAroundUnnamedInclude
#ifndef ClingWorkAroundMissingSmartInclude
#include "Event_3.cxx+"
#endif
void runClonesArrayEvo() {
#else
{
#endif
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L Event_3.cxx+");
#else
#ifndef ClingWorkAroundUnnamedInclude
   #include "Event_3.cxx+"
#endif
#endif
   TFile *f = TFile::Open("Event_2.root");
   TTree *t; f->GetObject("T",t);
   t->GetEntry(0);
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
