/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifdef __hpux
#include <iostream.h>
#else
#include <iostream>
using namespace std;
#endif

#ifdef __CINT__
#include <ertti.h>
#include "../inc/G__ci.h"
#else
#ifndef compiled
#include "Api.h"   // -I$CINTSYSDIR/src -I$CINTSYSDIR
#endif
#endif

//////////////////////////////////////////////////////////////
// FunctionObject for double f(int*,double)
//////////////////////////////////////////////////////////////
class FunctionObject {
#ifndef compiled
  G__ClassInfo globalscope;
  G__MethodInfo method;
  G__CallFunc func;
  long dummy;
  void *regeneratedp2f;
  int mode;
#else
  double (*regeneratedp2f)(int*,double);
#endif
 public:
    void Init(void *p2f) {
#ifndef compiled
       char *fname;
       // reconstruct function name
       fname=G__p2f2funcname(p2f);
       if(fname) {
          // resolve function overloading
          method=globalscope.GetMethod(fname,"int*,double",&dummy);
          if(method.IsValid()) {
             // get pointer to function again after overloading resolution
             regeneratedp2f=method.PointerToFunc();
             // check what kind of pointer is it
             mode = G__isinterpretedp2f(regeneratedp2f);
             switch(mode) {
                case G__INTERPRETEDFUNC: // reconstruct function call as string
                   break;
                case G__BYTECODEFUNC: // calling bytecode function
                   func.SetBytecode((struct G__bytecodefunc*)regeneratedp2f);
                   break;
                case G__COMPILEDINTERFACEMETHOD: // using interface method
                   func.SetFunc((G__InterfaceMethod)regeneratedp2f);
                   break;
                case G__COMPILEDTRUEFUNC: // using true pointer to function
                   break;
                case G__UNKNOWNFUNC: // this case will never happen
                   break;
             }
          }
          else {
             cerr << "no overloading parameter matches" << endl;
          }
       }
       else {
          cerr << "unknown pointer to function" << endl;
       }
#else
       regeneratedp2f= (double (*)(int*,double))p2f;
#endif
    }

  double Do(int* a,double b) {
    double result;
#ifndef compiled
    char temp[200];
    switch(mode) {
    case G__INTERPRETEDFUNC: // reconstruct function call as string
      sprintf(temp,"%s((int*)%ld,%g)",(char*)regeneratedp2f,(long)a,b);
#ifdef __CINT__
      result=G__calc(temp);
#else
      result=G__double(G__calc(temp));
#endif
      break;
    case G__BYTECODEFUNC: // calling bytecode function
    case G__COMPILEDINTERFACEMETHOD: // using interface method
      func.ResetArg();
      func.SetArg((long)a);
      func.SetArg((double)b);
      result=func.ExecDouble((void*)NULL);
      break;
    case G__COMPILEDTRUEFUNC: // using true pointer to function
    case G__UNKNOWNFUNC: // this case will never happen
      double (*p)(int*,double) ;
      p = (double (*)(int*,double))regeneratedp2f;
      result=(*p)(a,b);
      break;
    default:
      result = 0;
      break;
    }
#else
    result=(*regeneratedp2f)(a,b);
#endif
    return result;
  }
};

//////////////////////////////////////////////////////////////
// Plug in functions
//////////////////////////////////////////////////////////////
double f1(int* a,double b) {
  double result = b*(*a);
  return result;
}

double f2(int* a,double b) {
  double result = b+(*a);
  return result;
}



//////////////////////////////////////////////////////////////
// main test
//////////////////////////////////////////////////////////////
void test(void* p2f,int n) {
  FunctionObject fx;
  fx.Init(p2f);
  double sum=0;
  int a;
  double b=0.1;

  for(int i=0;i<n;i++) {
    a = i;
    sum += fx.Do(&a,b);
  }
  cout << "sum = " << sum << endl;

}


