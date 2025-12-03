{
#ifdef ClingWorkAroundMissingDynamicScope
if (TFile *f = (TFile*)gROOT->ProcessLine("gFile")) {
#else
if (gFile!=0) {
#endif
  return 0;
}
}
 
