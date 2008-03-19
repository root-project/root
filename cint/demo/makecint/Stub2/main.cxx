/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// main.cxx ,  Interpreted source

#ifdef __CINT__

// for CINT
#ifndef G__WIN32
#ifndef COMPILED_H
#pragma include_noerr "compiled.dll" #on_err "compiled.h"
#endif
#endif
#ifndef STUB_H
#pragma include_noerr "stub.dll"  #on_err "stub.h"
#endif

#else

// for compiler
#include "stub.h"

#endif

#define NEWFEATURE
#define TEST

#ifdef TEST
/////////////////////////////////////////////////////////////////////////
// Interpreted class derives from Stub class
/////////////////////////////////////////////////////////////////////////
#ifdef NEWFEATURE
class Interpreted : public Stub1 { 
 public:
  Interpreted() { }
  virtual void publicFunc1() {
    printf("Interpreted::publicFunc1()\n");
  } 
  void publicFunc2() {
    printf("Interpreted::publicFunc2()\n");
    printf("publicData1=%d  protectedData1=%d\n",publicData1,protectedData1);
    protectedFunc1();
  }
};
#endif
#endif

/////////////////////////////////////////////////////////////////////////
// Body of Stub1 member function
/////////////////////////////////////////////////////////////////////////
// giving definition to Stub1 virtual function
void Stub1::publicFunc1() {
  printf("Stub1::publicFunc1()\n");
}

// giving definition to Stub1 member function
void Stub1::publicFunc2() {
  printf("Stub1::publicFunc2()\n");
  printf("publicData1=%d  protectedData1=%d\n",publicData1,protectedData1);
  protectedFunc1();
}

#ifdef __CINT__
// Non virtual member function can be freely added to Stub1 class
void Stub1::publicFunc3() {
  printf(">> Note: It is fine to see diff of following 3 lines\n");
  printf("Stub1::publicFunc3()\n");
  printf("publicData1=%d  protectedData1=%d\n",publicData1,protectedData1);
  protectedFunc1();
}
#endif

///////////////////////////////////////////////////////////////////////
// Interpreted Global functions
///////////////////////////////////////////////////////////////////////
void interpretedVirtualFuncCall(Compiled1** obj,int n) { 
  printf("---- Interpreted Virtual Function Call ---\n");
  for(int i=0;i<n;i++) {
    obj[i]->publicFunc1(); // Virtual function call for Compiled1 
  }
}

/////////////////////////////////////////////////////////////////////////
// Body of stub function
/////////////////////////////////////////////////////////////////////////
void interpretedGlobalStubFunc1() { 
#ifdef NEWFEATURE
  const int N=5;
#else
  const int N=4;
#endif

  Compiled1* obj[N];
  obj[0] = new Compiled1;
  obj[1] = new Stub1;
  obj[2] = new Stub1;
  obj[3] = new Compiled1;
#ifdef NEWFEATURE
  obj[4] = new Interpreted;
#endif
  compiledVirtualFuncCall(obj,N);
  interpretedVirtualFuncCall(obj,N);
}

#ifndef TEST
/////////////////////////////////////////////////////////////////////////
// Interpreted class derives from Stub class
/////////////////////////////////////////////////////////////////////////
#ifdef NEWFEATURE
class Interpreted : public Stub1 { 
 public:
  Interpreted() { }
  virtual void publicFunc1() {
    printf("Interpreted::publicFunc1()\n");
  } 
  void publicFunc2() {
    printf("Interpreted::publicFunc2()\n");
    printf("publicData1=%d  protectedData1=%d\n",publicData1,protectedData1);
    protectedFunc1();
  }
};
#endif
#endif


/////////////////////////////////////////////////////////////////////////
int main() {
  Stub1 x;
  printf("### compiled protected member access ###\n");
  x.publicFunc2();
#ifdef __CINT__
  x.publicFunc3();
#endif

  printf("### interpreted/compiled virtual function resolution ###\n");
  interpretedGlobalStubFunc1();
  callInterpretedFuncFromCompiledFunc();

#ifdef NEWFEATURE
  Interpreted y;
  y.publicFunc2();
#ifdef __CINT__
  y.publicFunc3();
#endif
#endif

  return 0;
}



