struct MyClass {
   bool mybool;
};

#include "TFile.h"

void boolUpdate() {
   TFile *file = new TFile("boolUpdate.root","UPDATE");
   MyClass *m = new MyClass;
   file->WriteObject(m,"myobject");
   file->Write();
}