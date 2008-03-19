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
  }
  else {
    belongingmethod=(G__MethodInfo*)NULL;
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__MethodArgInfo::Name()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc;
    /* long property=0; */
    ifunc = G__get_ifunc_internal((struct G__ifunc_table*)belongingmethod->handle);
    return(ifunc->param[belongingmethod->index][argn]->name); 
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__MethodArgInfo::Property()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc;
    long property=0;
    ifunc = G__get_ifunc_internal((struct G__ifunc_table*)belongingmethod->handle);
    if(isupper(ifunc->param[belongingmethod->index][argn]->type)) 
      property|=G__BIT_ISPOINTER;
    if(ifunc->param[belongingmethod->index][argn]->pdefault) 
      property|=G__BIT_ISDEFAULT;
    if(ifunc->param[belongingmethod->index][argn]->reftype) 
      property|=G__BIT_ISREFERENCE;
    if(ifunc->param[belongingmethod->index][argn]->isconst&G__CONSTVAR) 
      property|=G__BIT_ISCONSTANT;
    if(ifunc->param[belongingmethod->index][argn]->isconst&G__PCONSTVAR) 
      property|=G__BIT_ISPCONSTANT;
    return(property);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
char* Cint::G__MethodArgInfo::DefaultValue()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc;
    /* long property=0; */
    ifunc = G__get_ifunc_internal((struct G__ifunc_table*)belongingmethod->handle);
    return(ifunc->param[belongingmethod->index][argn]->def); 
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
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc;
    ifunc = G__get_ifunc_internal((struct G__ifunc_table*)belongingmethod->handle);
    type.type = ifunc->param[belongingmethod->index][argn]->type;
    type.tagnum=ifunc->param[belongingmethod->index][argn]->p_tagtable;
    type.typenum =ifunc->param[belongingmethod->index][argn]->p_typetable;
    type.reftype = ifunc->param[belongingmethod->index][argn]->reftype;
    type.class_property=0;
    type.isconst = ifunc->param[belongingmethod->index][argn]->isconst;
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
