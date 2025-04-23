{
// Fill out the code of the actual test
gROOT->ProcessLine(".L ToplevelClass.C+");

#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("Bool_t result;");
#else
Bool_t result;
#endif
   
gROOT->ProcessLine(".L Toplevel.C");

#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(
   "result = WriteToplevel();"
   "result &= ReadToplevel();"
                   );
#else
result = WriteToplevel();
result &= ReadToplevel();
#endif

#ifdef ClingWorkAroundMissingUnloading
   gROOT->ProcessLine(
                      "result = WriteToplevel();"
                      "result &= ReadToplevel();"
                      );
#else   
gROOT->ProcessLine(".U Toplevel.C");
gROOT->ProcessLine(".L Toplevel.C+");

result &= WriteToplevel();
result &= ReadToplevel();
#endif

#ifdef ClingWorkAroundMissingDynamicScope
   gApplication->Terminate(gROOT->ProcessLine("!result"));
#else
   return !result; // invert value for Makefile purpose
#endif
}
