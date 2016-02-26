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
fprintf(stderr,"Executed testSelector ctor from JITed code\n");
fprintf(stderr,"testSelector result is 1 524288 1\n");
fprintf(stderr,"Info in <ACLiC>: script has already been loaded in interpreted mode\n");
fprintf(stderr,"Info in <ACLiC>: unloading testSelector.C and compiling it\n");
#endif
gROOT->ProcessLine(".L testSelector.C+");
#if defined(ClingWorkAroundMissingDynamicScope) || defined(ClingWorkAroundBrokenUnnamedReturn)
bool resInterp;
resInterp = gROOT->ProcessLine("runtest();");
resInterp = !resInterp;
#else
bool res = runtest();
return !res;
#endif
}
