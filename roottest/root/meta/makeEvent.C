void makeEvent(Int_t nevent = 400,Int_t comp = 1, Int_t split = 1,
               Int_t arg4 = 1, Int_t arg5 = 600) 
{
   gROOT->ProcessLine(".L Event.cxx+");
   gROOT->ProcessLine(".L MainEvent.cxx+");
#if defined(ClingWorkAroundMissingDynamicScope) || defined(__CLING__)
   gROOT->ProcessLine(TString::Format("MainEvent(%d,%d,%d,%d,%d);",nevent,comp,split,arg4,arg5));
#else
   MainEvent(nevent,comp,split,arg4,arg5);
#endif
}

