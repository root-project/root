{
if (!gSystem->CompileMacro("Data.cxx","k")) gApplication->Terminate(1);
if (!gSystem->CompileMacro("fillTree.cxx","k")) gApplication->Terminate(1);
fillTree();
}
