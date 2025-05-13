#include "TFile.h"
#include <iostream>

void runCloseTWebFile()
{
   // Make sure there is no crash when quitting Root after closing a TWebFile

   auto f = TFile::Open("https://root.cern/files/na49.root");
   if (f) {
      f->Close();
      // deleting the file was solving the problem, but not deleting the file
      // should not lead to a segfault when quitting Root anyway...
      // delete f;
   }
   else {
      std::cerr << "failed to open https://root.cern/files/na49.root" << std::endl;
   }
}
