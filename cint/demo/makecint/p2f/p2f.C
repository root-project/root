/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <iostream.h>

#ifdef __CINT__
//#error "This source must be compiled"
#include <ertti.h>
#include <G__ci.h>
#else
#include "Api.h"   // -I$CINTSYSDIR/src -I$CINTSYSDIR
#endif

#include "p2f.h"

void ExecP2F::SetP2F(void *p) {
  // reconstruct function name
  fname=G__p2f2funcname(p);
  if(fname) {
    G__ClassInfo globalscope;
    G__MethodInfo method;
    long dummy;
    // resolve function overloading
    method=globalscope.GetMethod(fname,"char*,double",&dummy);
    if(method.IsValid()) {
      // int (*p)(char*,double);
      char temp[200];
      int result;
      // get pointer to function again after overloading resolution
      p2f=method.PointerToFunc();
      mode = G__isinterpretedp2f(p2f);
    }
    else {
      p2f=(void*)NULL;
      mode = G__UNKNOWNFUNC;
    }
  }
  else {
    p2f=(void*)NULL;
    mode = G__UNKNOWNFUNC;
  }
}

int ExecP2F::DoP2F(char* s,double d) {
  char temp[200];
  int result;
  G__CallFunc func;
  int (*p)(char*,double);

  /* pointer to function is not set */
  if(!p2f) return(0);

  /* exec pointer to function */
  switch(mode) {
  case G__INTERPRETEDFUNC: // reconstruct function call as string
    sprintf(temp,"%s(%ld,%g)",fname,s,d);
#ifdef __CINT__
    result=G__calc(temp);
#else
    result=G__int(G__calc(temp));
#endif
    break;
  case G__BYTECODEFUNC: // calling bytecode function
    func.SetBytecode((struct G__bytecodefunc*)p2f);
    func.SetArg((long)s);
    func.SetArg((double)d);
    result=func.ExecInt((void*)NULL);
    break;
  case G__COMPILEDINTERFACEMETHOD: // using interface method
    func.SetFunc((G__InterfaceMethod)p2f);
    func.SetArg((long)s);
    func.SetArg((double)d);
    result=func.ExecInt((void*)NULL);
    break;
  case G__COMPILEDTRUEFUNC: // using true pointer to function
    p = (int (*)(char*,double))p2f;
    result=(*p)(s,d);
    break;
  case G__UNKNOWNFUNC: // this case will never happen
    p = (int (*)(char*,double))p2f;
    result=(*p)(s,d);
    break;
  }
  return(result);
}

int CompiledFunc(char* s,double d) {
  int result;
  result = sprintf(s,"CompiledFunc %g\n",d);
  return(result);
}
