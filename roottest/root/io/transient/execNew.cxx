#include "TFile.h"
#include "TClass.h"

class MyClass {
   int fOld; // !
   int fKeep;
public:
   MyClass() : fOld(0),fKeep(0) {};
   MyClass(int v): fOld(2*v),fKeep(3*v) {};
   void Print() const {
      fprintf(stdout,"fOld = %d, fKeep = %d\n",fOld,fKeep);
   }
   virtual ~MyClass() {}
   ClassDef(MyClass,3);
};

void write() {
   TFile *f = TFile::Open("trans.root","RECREATE");
   MyClass m(3);
   f->WriteObject(&m,"m");
   delete f;
}

void read() {
   TFile *f = TFile::Open("trans.root");
   if (f) {
      MyClass *m = 0;
      f->GetObject("m",m);
      m->Print();
   } else {
      fprintf(stderr,"Can not find trans.root\n");
   }
}

void execNew() {
   read();
}

#ifdef __MAKECINT__
#pragma read sourceClass="MyClass" targetClass="MyClass" source="int fOld" \
         version="[1-]" target="fOld" \
         code="{ fOld = onfile.fOld; }"
#endif 
