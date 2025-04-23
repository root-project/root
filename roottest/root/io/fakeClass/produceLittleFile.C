{
#ifndef SECOND_RUN
gROOT->ProcessLine(".L little.C+");
#endif
#if defined(ClingWorkAroundMissingDynamicScope)  && !defined(SECOND_RUN)
#define SECOND_RUN
   gROOT->ProcessLine(".x produceLittleFile.C");
#else
   
wrapper *e = new wrapper;
TFile *file = new TFile("little.root","RECREATE");
e->Write();
file->Write();
delete file;
#endif
}
