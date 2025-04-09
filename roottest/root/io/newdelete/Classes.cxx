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
  test01( new ROOT7014_class("abc") );

  test02( new TStruct );
  test02( new TNodefault(10) );
  test02( new TNodefaultOpNew(11) );
  test02( new TDefault(33) );
  test02( new TPartialDefault(TObject::Class()) );
  test02( new TNormal(44) );
  // test02( new ROOT7014_class("abc") ); no ClassDef

  
  TNormal m;
  m.s = new TPrivateDefaultConstructor(1);
  TFile file("sock.root","RECREATE");
  m.s->Write();
  m.Write();
  file.Write();
  file.Close();
  gSystem->Unlink("sock.root");

  return true;
}
