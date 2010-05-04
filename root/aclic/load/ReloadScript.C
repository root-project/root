#include <TTree.h>
#include <TFile.h>
#include <iostream>
#include <vector>
#include <string>

void reloadScript(const char* name) {
  TFile f(name, "READ");
  f.Close();
}


