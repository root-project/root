/*
 * Compile and link with:
 *
 * g++ -Wall -pedantic -Werror $(root-config --cflags) -L"$(root-config --libdir)" -lGeom -lCore -o readGeometry.exe readGeometry.cpp
 *
 *
 */

#include "TGeoManager.h"
#include "TClass.h"

#include <iostream>

int doParse(const char *GDMLpath, const char* OutputFile) {

   auto* pManager = TGeoManager::Import(GDMLpath);

   if (OutputFile) {
      std::cout << "Writing the content back into '" << OutputFile << "'."
      << std::endl;
      pManager->Export(OutputFile);
   }
   
   return 0;
}

int execParse(const char *GDMLpath = "dune10kt_v1_workspace.gdml",
              const char* OutputFile = nullptr)
{
   if (TClass::GetClass("TGDMLParse")==nullptr) {
      std::cout << "No gdml parser available\n";
      return 0;
   }
   return doParse(GDMLpath,OutputFile);
}

int main(int argc, char** argv) {

   if (argc < 2 || argc > 3) {
      std::cerr << "Usage:  " << argv[0] << "  GDMLpath [OutputFile]" << std::endl;
      return 1;
   }

   const char* GDMLpath = argv[1];
   const char* OutputFile = (argc > 2)? argv[2]: nullptr;

   return doParse(GDMLpath,OutputFile);
} // int main()
