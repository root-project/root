{
  gROOT->ProcessLine(".L writeFile.C+");
  writeFile();
  readFile();
  return 0;
}
