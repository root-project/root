#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

void runscript(const char *fname, bool with_rootmap = false)
{
   if (with_rootmap) {
      int old = gInterpreter->SetClassAutoloading(kFALSE);
      gInterpreter->LoadLibraryMap("libbtag.rootmap");
      gInterpreter->LoadLibraryMap("libjet.rootmap");
      gInterpreter->LoadLibraryMap("libsjet.rootmap");
      gInterpreter->SetClassAutoloading(old);
   }

   std::ifstream f(fname);
   if (!f) {
      std::cout << "Not able to open " << fname << std::endl;
      return;
   }

   std::string line;
   while (std::getline(f, line)) {
      // std::cout << "Process: " << line << std::endl;
      gROOT->ProcessLine(line.c_str());
   }
}

