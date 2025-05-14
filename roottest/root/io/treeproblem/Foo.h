// -*- mode: c++ -*-
//
// $Id$
// $Author$
// $Date$
// $Copyright: (C) 2002 BRAHMS Collaboration <brahmlib@rhic.bnl.gov>
//
#ifndef TREEPROBLEM_Foo
#define TREEPROBLEM_Foo

#include "TObject.h"

class Foo : public TObject
{
private:
  Int_t fFoo;
public:
  Foo() : TObject(), fFoo(0) {}
  Foo(Int_t foo) : TObject(), fFoo(foo) {}
  Foo(const Foo& foo) : TObject(foo) { fFoo = foo.GetFoo(); }
  virtual ~Foo() {}

  void  SetFoo(Int_t foo=0) { fFoo = foo; }
  Int_t GetFoo() const { return fFoo; }
  void  Print(Option_t * = "") const;

  ClassDef(Foo,1) // DOCUMENT ME
};

#endif
//____________________________________________________________________
//
// $Log$
//
