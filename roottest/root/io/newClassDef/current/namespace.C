#include "namespace.h"
#include "TBufferFile.h"

#ifndef __CINT__
//namespace MySpace {
  ClassImp(MySpace::A)
  ClassImp(MySpace::MyClass)
     //}
#endif

using namespace MySpace;

static MyClass nested0(1);
static MyClass *pnested0 = 0;

void testNamespaceWrite() {
  TFile * file = new TFile("test.root","RECREATE");
  MyClass obj;
  obj.Write("myclassobj");
  file->Write();
  file->Close();
  delete file;
}
  
TBuffer* n_writetest() 
{
  TBuffer *b = new TBufferFile(TBuffer::kWrite);
  *b << &nested0;
  return b;
}
void n_readtest(TBuffer & b) 
{
  b>>pnested0;
  if (pnested0->a!=nested0.a) {
     fprintf(stderr,"Error: MySpace::MyClass  not read properly!");
     fprintf(stderr,"Expected %d and got %d\n", 
             nested0.a,
             pnested0->a);
  }
}

void namespace_driver() {
  TBuffer *buf = n_writetest();
  buf->SetReadMode();
  buf->Reset();
  n_readtest(*buf);
  testNamespaceWrite();
}
