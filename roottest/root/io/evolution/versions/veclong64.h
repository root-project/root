#include <vector>
#include <iostream>

class MyClass {
public:
#if VERSION==1
   std::vector<long> fVec;
#elif VERSION==2
   std::vector<long long> fVec;
#endif
   void Print() {
      std::cout << "Print a vector of long";
#if VERSION==2
      std::cout << " long";
#endif
      std::cout << "\n";
      for(unsigned int i=0; i<fVec.size(); ++i) {
         std::cout << '#' << i << ' ' << std::hex << fVec[i] << '\n';
      }
   }
};

#include "TString.h"
#include "TFile.h"

int writefile(const char *prefix = "veclong") {
   TString filename;
   filename.Form("%s-%d.root",prefix,VERSION);
   TFile *f = new TFile(filename,"RECREATE");
   MyClass c;
#if VERSION==1
   c.fVec.push_back( (long)( 1LL << 40 | 0x3 ) );
#else
   c.fVec.push_back( 1LL << 40 | 0x3 );
#endif
   c.Print();
   f->WriteObject(&c,"veclong");
   delete f;
   return 0;
}

int readfile(const char *prefix = "veclong") {
   for(int i=1; i<3; ++i) {
      TString filename;
      filename.Form("%s-%d.root",prefix,i);
      TFile *f = new TFile(filename,"READ");
      if (f==0) continue;
      MyClass *c;
      f->GetObject("veclong",c);
      if (c) {
         c->Print();
      }
   }
   return 0;
}

int write_what(const char * /* version */) {
   return writefile();
}

#ifdef __MAKECINT__
#pragma link C++ class MyClass+;
#pragma link C++ function writefile;
#pragma link C++ function readfile;
#pragma link C++ function write_what;
#endif
   
