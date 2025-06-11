#include "TFile.h"
#include <iostream>

#if ARR == 1
const int arrsize = 5;
#elif ARR == 2
const int arrsize = 3;
#else
const int arrsize = 7;
#endif

class One {
public:
   int myarr[arrsize];
   One() { for(int i = 0; i < arrsize; ++i) myarr[i] = 0; }
   One(int start)  { for(int i = 0; i < arrsize; ++i) myarr[i] = start + i; }
   void Print() {
      for(int i = 0; i < arrsize; ++i) {
         std::cout << i << " " << myarr[i] << std::endl;
      };
   }
};

void write(const char *filename = "arr.root")
{
   One a(arrsize);
   auto f = TFile::Open(filename, "RECREATE");
   f->WriteObject(&a,"myone");
   delete f;
};

void read(const char *filename = "arr.root")
{
   One *a = nullptr;
   auto f = TFile::Open(filename);
   f->GetObject("myone", a);
   if (a) a->Print();
};

#ifdef __MAKECINT__
#pragma link C++ function write;
#pragma link C++ function read;
#pragma link C++ class One+;
#endif