{

if (!gSystem->CompileMacro("relations_load.C","k")) gApplication->Terminate(1);
if (!gSystem->CompileMacro("relations_write.C","k")) gApplication->Terminate(1);
if (!gSystem->CompileMacro("relations_read.C","k")) gApplication->Terminate(1);

#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("Write(true); Read(true);");
#else
   Write(true);
   Read(true);
#endif
}
