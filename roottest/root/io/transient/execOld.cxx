#include "TFile.h"
#include "TClass.h"

class MyClass {
   int fOld;
   int fKeep;
public:
   MyClass() : fOld(0),fKeep(0) {};
   MyClass(int v): fOld(2*v), fKeep(v) {};
   void Print() const {
      fprintf(stdout,"fOld = %d, fKeep = %d\n",fOld,fKeep);
   }
   virtual ~MyClass() {}
   ClassDef(MyClass,2);
};

void write() {
   TFile *f = TFile::Open("trans.root","RECREATE");
   MyClass m(3);
   f->WriteObject(&m,"m");
   delete f;
}

void read() {
   TFile *f = TFile::Open("trans.root");
   MyClass *m = 0;
   f->GetObject("m",m);
   m->Print();
}

void execOld() {
   write();
   read();
}
