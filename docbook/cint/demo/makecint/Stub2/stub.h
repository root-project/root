/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// stub.h , stub definition for partly compiled and partly interpreted class
#ifndef STUB_H
#define STUB_H

#ifdef __MAKECINT__
#pragma include_noerr "compiled.dll" #on_err "compiled.h"
#else
#include "compiled.h"
#endif

/////////////////////////////////////////////////////////////////////////
// Interface of Stub1 class is compiled so that 
//  - We can use this class in a compiled code
//  - We can resolve virtual function 
// However, body of member functions are interpreted.
/////////////////////////////////////////////////////////////////////////
class Stub1 : public Compiled1 { 
 public:
  Stub1() { }
  int publicData2;
  virtual void publicFunc1(); 
  void publicFunc2();

  long G__virtualinfo; // new feature, this allows you to inherit an 
                       // interpreted class from a Stub class
};

/////////////////////////////////////////////////////////////////////////
// interface of following function is compiled so that this can be used
// from compiled code. However, body of the function is interpreted.
/////////////////////////////////////////////////////////////////////////
void interpretedGlobalStubFunc1(); 

#endif
