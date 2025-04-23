#include "Riostream.h"
#include "TFile.h"

#if (ARR==1)
const int size = 5;
#elif (ARR==2)
const int size = 3;
#else
const int size = 7;
#endif

class One {
public:
   int myarr[size];
   One() { for(int i = 0; i < size; ++i) myarr[i] = 0; }
   One(int start)  { for(int i = 0; i < size; ++i) myarr[i] = start + i; }
   void Print() {
      for(int i = 0; i < size; ++i) {
         cout << i << " " << myarr[i] << endl;
      };
   }
};

void write(const char *filename = "arr.root") {
   One a(size);
   TFile *f = new TFile("arr.root","RECREATE");
   f->WriteObject(&a,"myone");
   delete f;
};

void read(const char *filename = "arr.root") {
   One *a;
   TFile *f = new TFile("arr.root");
   f->GetObject("myone",a);
   if (a) a->Print();
};

#ifdef __MAKECINT__
#pragma link C++ funtion write;
#pragma link C++ funtion read;
#pragma link C++ class One+;
#endif