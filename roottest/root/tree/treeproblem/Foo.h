// -*- mode: c++ -*-
//
// $Id$
// $Author$
// $Date$
// $Copyright: (C) 2002 BRAHMS Collaboration <brahmlib@rhic.bnl.gov>
//
#ifndef TREEPROBLEM_Foo
#define TREEPROBLEM_Foo
#ifndef ROOT_TObject
#include "TObject.h"
#endif

class Foo : public TObject
{
private:
  Int_t fFoo; 
public:
  Foo()  : fFoo(0) {}
  Foo(Int_t foo) : fFoo(foo) {}
  Foo(const Foo& foo) { fFoo = foo.GetFoo(); }
  ~Foo() override {}

  void  SetFoo(Int_t foo=0) { fFoo = foo; }
  Int_t GetFoo() const { return fFoo; }
  void  Print(Option_t* option="") const override;

  ClassDefOverride(Foo,1) // DOCUMENT ME
};

#endif
//____________________________________________________________________
//
// $Log$
//
