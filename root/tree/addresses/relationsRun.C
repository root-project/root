{

if (!gSystem->CompileMacro("relations_load.C","k")) gApplication->Terminate(1);
if (!gSystem->CompileMacro("relations_write.C","k")) gApplication->Terminate(1);
if (!gSystem->CompileMacro("relations_read.C","k")) gApplication->Terminate(1);

Write(true);
Read(true);

}
