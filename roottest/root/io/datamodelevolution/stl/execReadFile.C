{
  gROOT->ProcessLine(".L readFile.C+");
#ifdef ClingWorkAroundMissingDynamicScope
  gROOT->ProcessLine("readFile();readFileList();readFileSet();");
#else
  readFile();
  readFileList();
  readFileSet();
#endif
  return 0;
}
