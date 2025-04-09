{
gROOT->ProcessLine(".L testSel.C+");

// Insure that the library is not loaded instead of the 
// script
gInterpreter->UnloadLibraryMap("testSelector_C");

gROOT->ProcessLine(".L testSelector.C");
bool res = runtest();
if (!res) return !res;
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
