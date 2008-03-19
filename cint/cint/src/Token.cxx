/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Token.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~1998  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include "Api.h"
#include "common.h"


/*********************************************************************
* class G__TokenInfo
*
* Outcome of discussion between Nenad Buncic of CERN. 15 Mar 1997
* 
*********************************************************************/

///////////////////////////////////////////////////////////////////////////
void Cint::G__TokenInfo::Init() 
{
  // reset status of the object 
  tokentype= t_invalid;
  tokenproperty= p_invalid;
  methodscope.Init();
  // tinfo.Init();
  glob.Init();
  nextscope.Init();
  bytecode=(struct G__bytecodefunc*)NULL;
  localvar=(struct G__var_array*)NULL;
}

///////////////////////////////////////////////////////////////////////////
Cint::G__TokenInfo::G__TokenInfo(const G__TokenInfo& tki) :
  tokentype(tki.tokentype), 
  tokenproperty(tki.tokenproperty), 
  methodscope(tki.methodscope),
  bytecode(tki.bytecode),
  localvar(tki.localvar),
  glob(tki.glob),
  nextscope(tki.nextscope),
  tinfo(tki.tinfo)
{
  // Copy constructor
}

///////////////////////////////////////////////////////////////////////////
G__TokenInfo& Cint::G__TokenInfo::operator=(const G__TokenInfo& tki)
{

  // Assignment operator

  if(this!=&tki) {
    tokentype=tki.tokentype; 
    tokenproperty=tki.tokenproperty; 
    methodscope=tki.methodscope;
    bytecode=tki.bytecode;
    localvar=tki.localvar;
    glob=tki.glob;
    nextscope=tki.nextscope;
    tinfo=tki.tinfo;
  }
  return *this;
}

///////////////////////////////////////////////////////////////////////////
// MakeLocalTable has to be used when entering to a new function
G__MethodInfo Cint::G__TokenInfo::MakeLocalTable(G__ClassInfo& tag_scope
                                           ,const char* fname
					   ,const char* paramtype) 
{
  long dmy;

  Init();

  // get handle to function
  methodscope = tag_scope.GetMethod(fname,paramtype,&dmy);

  // need to set flag to proceed compilation even with problem
  // to be implemented

  // compile bytecode to get local table, method validity is checked inside
  bytecode = methodscope.GetBytecode(); 

  // reset the flag
  // to be implemented

  // reset method if bytecode compilation was not successful
  if(bytecode) {
    localvar=bytecode->var;
  }
  else {
    localvar=(struct G__var_array*)NULL;
    methodscope.Init();
  }

  return(methodscope);
}
///////////////////////////////////////////////////////////////////////////
// Query has to be used to get information for each token
int Cint::G__TokenInfo::Query(G__ClassInfo& tag_scope
			,G__MethodInfo& func_scope
			,const char* /* preopr */ ,const char* name
			,const char* postopr)
{
  nextscope.Init(); // initialize nesting scope information
  // search token matches in following order
  if(SearchTypeName(name,postopr))                  return(1);
  if(SearchLocalVariable(name,func_scope,postopr))  return(1);
  if(SearchDataMember(name,tag_scope,postopr))      return(1);
  if(SearchGlobalVariable(name,postopr))            return(1);
  if(SearchMemberFunction(name,tag_scope))          return(1);
  if(SearchGlobalFunction(name))                    return(1);
  // no match
  tokenproperty = p_invalid;
  tokentype = t_invalid;
  /* preopr = postopr; preopr not used, this statement isn't needed */
  return(0);
}
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// Private member functions
///////////////////////////////////////////////////////////////////////////
int Cint::G__TokenInfo::SearchTypeName(const char* name,const char* postopr)
{
  tinfo.Init(name);
  if(tinfo.IsValid()) {
    tokenproperty = p_type;
    if(tinfo.Property()&G__BIT_ISENUM) tokentype = t_enum;
    else if(tinfo.Property()&G__BIT_ISTAGNUM) {
      tokentype = t_class;
      // set nextscope if followed by :: operator
      if(strcmp(postopr,"::")==0) nextscope = tinfo; // assign to baseclass
    }
    else if(tinfo.Property()&G__BIT_ISTYPEDEF) tokentype = t_typedef;
    else if(tinfo.Property()&G__BIT_ISFUNDAMENTAL) tokentype = t_fundamental;
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TokenInfo::SearchLocalVariable(const char* name,G__MethodInfo& func_scope
				     ,const char* postopr)
{
  if(localvar && func_scope.IsValid()) {
    if(&func_scope != &methodscope) {
      G__fprinterr(G__serr,"Warning: Cint::G__TokenInfo::SearchLocalVariable() func scope changed without Cint::G__TokenInfo::MakeLocalTable()\n");
      return(0);
    }
    struct G__var_array *var;
    int ig15;
    var = localvar;
    while(var) {
      for(ig15=0;ig15<var->allvar;ig15++) {
        if(strcmp(name,var->varnamebuf[ig15])==0) {
	  tokenproperty = p_data;
          tokentype = t_local;
	  if(tolower(var->type[ig15])=='u' && -1!=var->p_tagtable[ig15] &&
	     (strcmp(postopr,".")==0||strcmp(postopr,"->")==0)) {
            // set nextscope if followed by . or -> operator
	    nextscope.Init(var->p_tagtable[ig15]);
          } 
          return(1);
	}
      }
      var=var->next;
    }
  }
  // no match
  return(0);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TokenInfo::SearchDataMember(const char* name,G__ClassInfo& tag_scope
				  ,const char* postopr)
{
  if(tag_scope.IsValid() && tag_scope.HasDataMember(name)) {
    tokenproperty = p_data;
    tokentype = t_datamember;
    if(strcmp(postopr,".")==0 || strcmp(postopr,"->")==0) { 
      // set nextscope if followed by . or -> operator
      GetNextscope(name,tag_scope);
    }
    return(1);
  }
  else {
    return(0);
  }
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__TokenInfo::SearchGlobalVariable(const char* name,const char* postopr)
{
  if(glob.HasDataMember(name)) {
    tokenproperty = p_data;
    tokentype = t_datamember;
    if(strcmp(postopr,".")==0 || strcmp(postopr,"->")==0) { 
      // set nextscope if followed by . or -> operator
      GetNextscope(name,glob);
    }
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TokenInfo::SearchMemberFunction(const char* name,G__ClassInfo& tag_scope)
{
  if(tag_scope.IsValid() && tag_scope.HasMethod(name)) {
    tokenproperty = p_func;
    tokentype = t_memberfunc;
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__TokenInfo::SearchGlobalFunction(const char* name)
{
  if(glob.HasMethod(name)) {
    tokenproperty = p_func;
    tokentype = t_globalfunc;
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
// set nextscope for scope stacking
void Cint::G__TokenInfo::GetNextscope(const char* name,G__ClassInfo& tag_scope)
{
  G__DataMemberInfo dt(tag_scope);
  // iterate on variable table
  while(dt.Next()) {
    // check if name matchs
    if(strcmp(name,dt.Name())==0) {
      G__TypeInfo *ti = dt.Type();
      // set nextscope if it is a class or struct or union
      if(ti->Property()&G__BIT_ISTAGNUM) nextscope = *ti; 
      return;
    }
  }
}
///////////////////////////////////////////////////////////////////////////

