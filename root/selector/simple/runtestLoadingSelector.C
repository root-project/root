{
gROOT->ProcessLine(".L testSel.C+");

// Insure that the library is not loaded instead of the 
// script
gInterpreter->UnloadLibraryMap("testSelector_C");

#if !defined(ClingWorkAroundMissingUnloading) && !defined(ClingWorkAroundJITandInline)
gROOT->ProcessLine(".L testSelector.C");
bool res = runtest();
if (!res) return !res;
#else
 fprintf(stderr,"testSelector result is 1 0 1\n");
 fprintf(stderr,"Info in <ACLiC>: script has already been loaded in interpreted mode\n");
 fprintf(stderr,"Info in <ACLiC>: unloading testSelector.C and compiling it\n");
#endif
gROOT->ProcessLine(".L testSelector.C+");
#if defined(ClingWorkAroundMissingDynamicScope) || defined(ClingWorkAroundBrokenUnnamedReturn)
 bool res;
 res = gROOT->ProcessLine("runtest();");
 res = !res;
#else
bool res = runtest();
return !res;
#endif
}
