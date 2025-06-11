struct MyClass {
   bool mybool;
};

#include "TFile.h"

void boolUpdate()
{
   TFile *file = TFile::Open("boolUpdate.root","UPDATE");
   MyClass *m = new MyClass;
   file->WriteObject(m, "myobject");
   file->Write();
   file->ls();
   delete file;
}