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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/



#ifndef G__CALLFUNC_H
#define G__CALLFUNC_H

#ifndef G__API_H
#include "Api.h"
#endif

namespace Cint {

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

  G__CallFunc(const G__CallFunc& cf)
#ifndef __MAKECINT__
   :pfunc(cf.pfunc),
    result(cf.result),
#ifdef G__ASM_WHOLEFUNC
    bytecode(cf.bytecode),
#endif
    method(cf.method),
    para(cf.para)
#endif /* __MAKECINT__ */
  {}

  G__CallFunc& operator=(const G__CallFunc& cf) {
    if (&cf != this) {
#ifndef __MAKECINT__
      pfunc=cf.pfunc;
      result=cf.result;
#ifdef G__ASM_WHOLEFUNC
      bytecode=cf.bytecode;
#endif
      method=cf.method;
      para=cf.para;
#endif /* __MAKECINT__ */
    }
    return *this;
  }

  void Init() ;

  enum MatchMode { ExactMatch=0, ConversionMatch=1 };
  void SetFunc(G__ClassInfo* cls,const char* fname,const char* args
	       ,long* poffset,MatchMode mode=ConversionMatch);
  void SetFuncProto(G__ClassInfo* cls,const char* fname,const char* argtype,long* poffset);
  // begin old interface
  void SetFunc(G__InterfaceMethod f);
  void SetFunc(G__MethodInfo m);
#ifdef G__ASM_WHOLEFUNC
  void SetBytecode(struct G__bytecodefunc* bc);
#endif
  int IsValid() { /* return(pfunc?1:0l; */ return(method.IsValid());}
  void SetArgArray(long *p,int narg= -1);
  void ResetArg() { para.paran=0; }
  void SetArg(long l) ;
  void SetArg(unsigned long ul) ;
  void SetArg(double d) ;
  void SetArgRef(long& l) ;
  void SetArgRef(double& d) ;
  void SetArg( G__value );
#ifdef G__NATIVELONGLONG
  void SetArg(G__int64 ll);
  void SetArg(G__uint64 ull);
  void SetArg(long double ld);
  void SetArgRef(G__int64& ll);
  void SetArgRef(G__uint64& ull);
  void SetArgRef(long double& ld);
#endif
  // end old interface

  G__value Execute(void *pobject );
  void Exec(void *pobject) { Execute(pobject); }
  long ExecInt(void *pobject) { return G__int(Execute(pobject)); }
  double ExecDouble(void *pobject) { return G__double(Execute(pobject)); }
#ifdef G__NATIVELONGLONG
  G__int64 ExecInt64(void *pobject) { return G__Longlong(Execute(pobject)); }
#endif

  G__InterfaceMethod InterfaceMethod() { return pfunc; }
  void SetArgs(const char* args);
  void SetArgs(const G__param &p);
  G__MethodInfo GetMethodInfo() { return method; }

 private:
  void SetFuncType();
#ifndef __MAKECINT__
  G__InterfaceMethod pfunc;
  G__value result;
#ifdef G__ASM_WHOLEFUNC
  struct G__bytecodefunc *bytecode;
#endif
  G__MethodInfo method;
  struct G__param para;
  int ExecInterpretedFunc(G__value* presult);
#endif /* __MAKECINT__ */
};

} // namespace Cint

using namespace Cint;
#endif
