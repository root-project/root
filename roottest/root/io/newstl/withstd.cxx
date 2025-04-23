#include "TFile.h"
#include "TROOT.h"
#include <string>
#include "Riostream.h"

template <class T> class something {};

#ifdef __MAKECINT__
#pragma link C++ class something<std::string>;
#endif

void withstd() {

   cout << "GetClass(\"something<string>\") " << (void*)gROOT->GetClass("something<string>") << endl;
   cout << "GetClass(\"something<std::string>\") " << (void*)gROOT->GetClass("something<std::string>") << endl;
   cout << "GetClass(\"something<string>\") " << (void*)gROOT->GetClass("something<string>") << endl;
     
   //TFile *file = new TFile("withstd.root","RECREATE");
   //
   //file->Write();
   //delete file;
}