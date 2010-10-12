/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Token.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef G__TOKENINFO_H
#define G__TOKENINFO_H 


#ifndef G__API_H
#include "Api.h"
#endif

namespace Cint {

class G__ClassInfo;
class G__MethodInfo;

/*********************************************************************
* class G__TokenInfo
*
* Outcome of discussion between Nenad Buncic of CERN. 15 Mar 1997
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__TokenInfo {
 public:
  enum G__TokenType { t_invalid                                   // p_invalid
                    , t_class , t_typedef, t_fundamental , t_enum    // p_type
                    , t_memberfunc, t_globalfunc                     // p_func
                    , t_datamember, t_local, t_global, t_enumelement // p_data
                    };
  enum G__TokenProperty {p_invalid , p_type , p_data, p_func};

  ~G__TokenInfo() {}
  G__TokenInfo() :
    tokentype(t_invalid), tokenproperty(p_invalid), methodscope(),
    bytecode(NULL), localvar(NULL), glob(), nextscope(), tinfo() { Init(); }
  G__TokenInfo(const G__TokenInfo& tki);
  G__TokenInfo& operator=(const G__TokenInfo& tki);
  void Init();

  // MakeLocalTable has to be used when entering to a new function
  G__MethodInfo MakeLocalTable(G__ClassInfo& tag_scope
                              ,const char* fname,const char* paramtype);

  // Query has to be used to get information for each token
  int Query(G__ClassInfo& tag_scope,G__MethodInfo& func_scope
	    ,const char* preopr,const char* name,const char* postopr);

  // Following functions have to be called after Query 
  enum G__TokenType GetTokenType() { return(tokentype); }
  enum G__TokenProperty GetTokenProperty() { return(tokenproperty); }
  G__ClassInfo GetNextScope() { return(nextscope); }

 private:
  enum G__TokenType tokentype; 
  enum G__TokenProperty tokenproperty; 
  G__MethodInfo methodscope;
  struct G__bytecodefunc *bytecode;
  struct G__var_array *localvar;
  G__ClassInfo glob;
  G__ClassInfo nextscope;
  G__TypeInfo tinfo;

  int SearchTypeName(const char* name,const char* postopr);
  int SearchLocalVariable(const char* name,G__MethodInfo& func_scope
			  ,const char* postopr);
  int SearchDataMember(const char* name,G__ClassInfo& tag_scope
		       ,const char* postopr);
  int SearchGlobalVariable(const char* name,const char* postopr);
  int SearchMemberFunction(const char* name,G__ClassInfo& tag_scope);
  int SearchGlobalFunction(const char* name);
  void GetNextscope(const char* name,G__ClassInfo& tag_scope);
};

} // namespace Cint

/*********************************************************************
* memo
*
*  int G__loadfile(char* fname);
*    #define G__LOADFILE_SUCCESS         0
*    #define G__LOADFILE_DUPLICATE       1
*    #define G__LOADFILE_FAILURE       (-1)
*    #define G__LOADFILE_FATAL         (-2)
*
*  int G__unloadfile(char* fname);
*    #define G__UNLOADFILE_SUCCESS    0
*    #define G__UNLOADFILE_FAILURE  (-1)
*
*  void G__add_ipath(char* pathname);
*
*  in src/Class.h
*  class G__ClassInfo {
*   public:
*    G__ClassInfo();
*    Init(char* classname);
*    int IsValid();
*    ..
*  };
*
*  in src/Method.h
*  class G__MethodInfo {
*   public:
*    G__MethodInfo();
*    G__MethodInfo(G__ClassInfo& scope);
*    Init();
*    Init(G__ClassInfo& scope);
*    int IsValid();
*    ..
*  };
* 
*********************************************************************/

using namespace Cint;
#endif
