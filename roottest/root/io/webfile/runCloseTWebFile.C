#include "TFile.h"
#include "Riostream.h"

void runCloseTWebFile()
{
   // Make sure there is no crash when quitting Root after closing a TWebFile 

   TFile *f = TFile::Open("http://root.cern.ch/files/na49.root");
   if (f) {
      f->Close();
      // deleting the file was solving the problem, but not deleting the file 
      // should not lead to a segfault when quitting Root anyway...
      // delete f;
   }
   else {
      cout << "failed to open http://root.cern.ch/files/na49.root" << endl;
   }
}
