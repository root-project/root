#include "namespace.h"

#ifndef __CINT__
#ifndef NOSPACE
namespace MySpace {
#endif
  ClassImp(MySpace::A);
  ClassImp(MyClass);
#ifndef NOSPACE
}
#endif
#endif

using namespace MySpace;

void testNamespaceWrite() {
  TFile * file = new TFile("test.root","RECREATE");
  MyClass obj;
  obj.Write("myclassobj");
  file->Write();
  //  file->Close();
};
  
