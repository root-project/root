/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file CallFunc.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/



#ifndef G__CALLFUNC_H
#define G__CALLFUNC_H

#include "Api.h"

/*********************************************************************
* class G__CallFunc
*
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__CallFunc {
 public:
  ~G__CallFunc() {}
  G__CallFunc() ;
  void Init() ;

#ifndef G__OLDIMPLEMENTATION2019
  enum MatchMode { ExactMatch=0, ConversionMatch=1 };
  void SetFunc(G__ClassInfo* cls,const char* fname,const char* args
	       ,long* poffset,MatchMode mode=ConversionMatch);
#else
  void SetFunc(G__ClassInfo* cls,const char* fname,const char* args,long* poffset);
#endif
#ifndef G__OLDIMPLEMENTATION540
  void SetFuncProto(G__ClassInfo* cls,const char* fname,const char* argtype,long* poffset);
#endif
  // begin old interface
  void SetFunc(G__InterfaceMethod f);
#ifndef G__OLDIMPLEMENTATION1749
  void SetFunc(G__MethodInfo m);
#endif
#ifdef G__ASM_WHOLEFUNC
  void SetBytecode(struct G__bytecodefunc* bc);
#endif
#ifndef G__OLDIMPLEMENTATION533
  int IsValid() { /* return(pfunc?1:0l; */ return(method.IsValid());}
  void SetArgArray(long *p,int narg= -1);
#endif
  void ResetArg() { para.paran=0; }
  void SetArg(long l) ;
  void SetArg(double d) ;
#ifdef G__NATIVELONGLONG
  void SetArg(G__int64 ll);
  void SetArg(G__uint64 ull);
  void SetArg(long double ld);
#endif
  // end old interface

  void Exec(void *pobject) ;
  long ExecInt(void *pobject) ;
  double ExecDouble(void *pobject) ;

#ifndef G__FONS50
  G__InterfaceMethod InterfaceMethod() { return pfunc; }
  void SetArgs(const char* args);
  G__MethodInfo GetMethodInfo() { return method; }
#endif

 private:
#ifndef G__OLDIMPLEMENTATION1591
  void SetFuncType();
#endif
#ifndef __MAKECINT__
  G__InterfaceMethod pfunc;
  G__value result;
#ifdef G__ASM_WHOLEFUNC
  struct G__bytecodefunc *bytecode;
#endif
#ifndef G__OLDIMPLEMENTATION533
  G__MethodInfo method;
#endif
  struct G__param para;
  int ExecInterpretedFunc(G__value* presult);
#endif /* __MAKECINT__ */
};


#endif
