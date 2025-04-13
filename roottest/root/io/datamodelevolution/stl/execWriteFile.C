{
  gROOT->ProcessLine(".L writeFile.C+");
#ifdef ClingWorkAroundMissingDynamicScope
  gROOT->ProcessLine("writeFile();readFile();");
#else
  writeFile();
  readFile();
#endif
  return 0;
}
