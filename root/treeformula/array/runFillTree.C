{
if (!gSystem->CompileMacro("Data.cxx","kf")) gApplication->Terminate(1);
if (!gSystem->CompileMacro("fillTree.cxx","kf")) gApplication->Terminate(1);
fillTree();
}
