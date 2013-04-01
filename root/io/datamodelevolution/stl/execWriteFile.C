{
  gROOT->ProcessLine(".L writeFile.C+");
  writeFile();
  readFile();
}
