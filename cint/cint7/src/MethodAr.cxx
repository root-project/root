/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file MethodAr.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"
#include "fproto.h"

using namespace Cint::Internal;

/*********************************************************************
* class G__MethodArgInfo
*
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodArgInfo::Init(class G__MethodInfo &a)
{
  if(a.IsValid()) {
    belongingmethod = &a;
    argn = -1;
    m_name = "";
  }
  else {
    belongingmethod=(G__MethodInfo*)NULL;
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__MethodArgInfo::Name()
{
   if(IsValid()) {
      if (m_name.length()==0) {
         m_name = belongingmethod->fFunc.FunctionParameterNameAt(argn).c_str();
      }
      return m_name.c_str();
   }
   else {
      return((char*)NULL);
   }
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__MethodArgInfo::Property()
{
   if(IsValid()) {
      long property=0;
      ::Reflex::Member func =  belongingmethod->fFunc;
      if(isupper(G__get_type(func.TypeOf().FunctionParameterAt(argn)))) 
         property|=G__BIT_ISPOINTER;
      if(func.FunctionParameterDefaultAt(argn).size()) 
         property|=G__BIT_ISDEFAULT;
      if(func.TypeOf().FunctionParameterAt(argn).FinalType().IsReference()) 
         property|=G__BIT_ISREFERENCE;
      if(G__test_const(func.TypeOf().FunctionParameterAt(argn),G__CONSTVAR))
         property|=G__BIT_ISCONSTANT;
      if(G__test_const(func.TypeOf().FunctionParameterAt(argn),G__PCONSTVAR)) 
         property|=G__BIT_ISPCONSTANT;
      return(property);
   }
   else {
      return(0);
   }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__MethodArgInfo::DefaultValue()
{
  if(IsValid()) {
    return(belongingmethod->fFunc.FunctionParameterDefaultAt(argn).c_str()); 
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodArgInfo::IsValid()
{
  if(belongingmethod && belongingmethod->IsValid()) {
    if(0<=argn&&argn<belongingmethod->NArg()) {
      return(1);
    }
    else {
      return(0);
    }
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodArgInfo::Next()
{
  ++argn;
  m_name = "";
  if(IsValid()) {
    type.Init( belongingmethod->fFunc.TypeOf().FunctionParameterAt(argn) );
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
