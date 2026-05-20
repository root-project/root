class MyClass {};

#include "TFile.h"
void RunMyClass() {
   TFile *f = new TFile("myclass.root","RECREATE");
   MyClass m;
   f->WriteObject(&m,"object");
}