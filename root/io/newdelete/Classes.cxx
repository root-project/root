#include "Classes.h"
#include "TClass.h"
#include "TROOT.h"
#include "TFile.h"
#include "TSystem.h"

template <class T> void test01(T *built) {
  TClass *cl;

  cl = gROOT->GetClass(typeid(T));

  T *st = (T*)cl->New();
  delete st;

  cl->Destructor(built);
}

template <class T> void test02(T *built) {
  TClass *cl;

  cl = gROOT->GetClass(typeid(T));

  T *st = (T*)cl->New();
  if (st) st->IsA()->Print();
  delete st;

  built->IsA()->Print();
  cl->Destructor(built);
}

bool Classes() {

  test01( new Struct );
  test01( new Nodefault(10) );
  test01( new NodefaultOpNew(11) );
  test01( new Default(33) );
  test01( new PartialDefault(TObject::Class()) );
  test01( new Normal(44) );

  test01( new TStruct );
  test01( new TNodefault(10) );
  test01( new TNodefaultOpNew(11) );
  test01( new TDefault(33) );
  test01( new TPartialDefault(TObject::Class()) );
  test01( new TNormal(44) );

  
  TNormal m;
  m.s = new TSocket("localhost","telnet");
  TFile file("sock.root","RECREATE");
  m.s->Write();
  m.Write();
  file.Write();
  file.Close();
  gSystem->Unlink("sock.root");

  return true;
}
