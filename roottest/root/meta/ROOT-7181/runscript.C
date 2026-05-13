#include <fstream>
#include <iostream>
#include <string>

void runscript(const std::string &fname, bool with_rootmap = false)
{
   if (with_rootmap) {
      int old = gInterpreter->SetClassAutoloading(kFALSE);
      gInterpreter->LoadLibraryMap("libbtag.rootmap");
      gInterpreter->LoadLibraryMap("libjet.rootmap");
      gInterpreter->LoadLibraryMap("libsjet.rootmap");
      gInterpreter->SetClassAutoloading(old);
   }

   std::ifstream f(fname);

   std::string str;
   while (std::getline(f, str)) {

      if (gSystem->InheritsFrom("TWinNTSystem"))
         if (str.find(".L") == 0) {
            auto p = str.find(".so");
            if (p != std::string::npos)
               str = str.substr(0, p) + ".dll";
         }

      if (str.length() > 0)
         gInterpreter->ProcessLineSynch(str.c_str());
   }
}
