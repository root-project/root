/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file bc_parse.h
 ************************************************************************
 * Description:
 *  block scope parser and compiler
 ************************************************************************
 * Copyright(c) 2004~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef BC_PARSE_H
#define BC_PARSE_H

#if !defined(__sun) && (!defined(_MSC_VER) || _MSC_VER > 1200) && !(defined(__xlC__) || defined(__xlc__))
//extern "C" {
#ifdef __CINT__
#include "../G__ci.h"
#else
#include "common.h"
#endif
//}
#else
#include "G__ci.h"
#include "common.h"
#endif

#include "bc_inst.h"
#include "bc_type.h"
#include "bc_reader.h"

#include <map>
#include <vector>
#include <deque>
#include <string>
using namespace std;

typedef int G__bc_pointer;
typedef int G__bc_pointer_addr;


////////////////////////////////////////////////////////////////////
// Virtual base class initialization
////////////////////////////////////////////////////////////////////
extern "C" void G__bc_Baseclassctor_vbase(int tagnum);

////////////////////////////////////////////////////////////////////
// Generate instruction
////////////////////////////////////////////////////////////////////
extern "C" void G__bc_VIRTUALADDSTROS(int tagnum
				      ,struct G__inheritance* baseclas
				      ,int basen) ;
extern "C" void G__bc_cancel_VIRTUALADDSTROS() ;
extern "C" void G__bc_REWINDSTACK(int n) ;
extern "C" int G__bc_casejump(void* p,int val);

////////////////////////////////////////////////////////////////////
// G__bc_new_operator
////////////////////////////////////////////////////////////////////
extern "C" G__value G__bc_new_operator(const char *expression) ;
extern "C" void G__bc_delete_operator(const char *expression,int isarray) ;

////////////////////////////////////////////////////////////////////
// G__Isvalidassignment
////////////////////////////////////////////////////////////////////
extern "C" int G__Isvalidassignment(G__TypeReader& ltype,G__TypeReader& rtype,G__value* rval);
extern "C" int G__Isvalidassignment_val(G__value* ltype,int varparan,int lparan,int lvar_type,G__value* rtype);
extern "C" int G__bc_conversion(G__value *result
				,struct G__var_array* var,int ig15
				,int var_type,int paran) ;

/***********************************************************************
 * G__jumptable
 ***********************************************************************/
class G__breaktable {
  vector<int> m_breaktable;
 public:
  void add(int originpc) { m_breaktable.push_back(originpc); }
  void resolve(G__bc_inst& inst,int destination); 
};

/***********************************************************************
 * G__casetable 
 ***********************************************************************/
class G__casetable {
  map<long,long> m_casetable;
  int            m_default;
 public:
  void addcase(int caseval,int destinationpc) {m_casetable[caseval]=destinationpc;}
  void adddefault(int defaultpc) { m_default = defaultpc; }
  //void resolve() { /* do nothing here */ }
  int jump(int val) ;
};

/***********************************************************************
 * G__labeltable 
 ***********************************************************************/
class G__gototable {
  map<string,int> m_labeltable;
  map<string,int> m_gototable;
 public:
  void clear() { m_labeltable.clear(); m_gototable.clear(); }
  void addlabel(const string& label,int labelpc) {m_labeltable[label]=labelpc;}
  void addgoto(int originpc,const string& label) {m_gototable[label]=originpc;}
  void resolve(G__bc_inst& inst);
};

/***********************************************************************
 * G__blockscope
 ***********************************************************************/
class G__blockscope {
  friend class G__blockscope_expr;
 protected:
  ////////////////////////////////////////////////////////////////////
  // function ID, moved from G__compiler
  ////////////////////////////////////////////////////////////////////
  struct G__ifunc_table *m_ifunc;
  int m_iexist;

  struct G__var_array    *m_var;
  struct G__var_array    *store_p_local;

  // reader object is allocated and deleted in G__functionblock or
  // G__unnamedmacro. Those classes inherit from G__blockscope. 
  // We should not assign,modify or delete m_preader in G__blockscope.
  G__virtualreader *m_preader;

  G__bc_inst      m_bc_inst;

  int isvirtual;
  int isstatic;
  G__casetable    *m_pcasetable;
  G__breaktable   *m_pbreaktable;
  G__breaktable   *m_pcontinuetable;
  G__gototable    *m_pgototable;

  void setcasetable(G__casetable* x) { m_pcasetable=x; }
  void setbreaktable(G__breaktable* x) { m_pbreaktable=x; }
  void setcontinuetable(G__breaktable* x) { m_pcontinuetable=x; }
  void setgototable(G__gototable* x) { m_pgototable=x; }

 public:
  G__blockscope();
  G__blockscope(G__blockscope* enclosing);
  ~G__blockscope();

  void Init(G__blockscope* enclosing=0);

  // read a token and separator
  // int fgettoken(string& token);

  // parsing 
  // top entry for compilation
  int compile(int openBrace=0);
  int compile_core(int openBrace=0);

  // called from G__bc_new_operator
  G__value compile_newopr(const string& expression);
  void compile_deleteopr(string& expression,int isarray);

  // Initialize virtual base class. Moved from G__functionscope
  void Baseclassctor_vbase(int tagnum);

  // access rule check
  int access(/* const */ G__MethodInfo& x) const;
  int access(/* const */ G__DataMemberInfo& x) const;

  int access(int tagnum,long property) const;
  int isfriend(int tagnum) const;

  G__bc_inst& GetInst() { return(m_bc_inst); }
  int GetTagnum() const { return(m_ifunc->tagnum); }

 private:
  // operator
  int compile_space(string& token,int c);
  int compile_case(string& token,int c);
  int compile_default(string& token,int c);
  int compile_operator(string& token,int c);
  int compile_operator_PARENTHESIS(string& token,int c);
  int compile_operator_AND_ASTR(string& token,int c);
  int compile_operator_LESS(string& token,int c);
  int compile_operator_DIV(string& token,int c);
  int compile_bracket(string& token,int c);
  int compile_column(string& token,int c);
  int compile_semicolumn(string& token,int c);
  //int compile_quotation(string& token,int c);
  int compile_parenthesis(string& token,int c);
  int compile_brace(string& token,int c);
  int compile_new(string& token,int c);
  int compile_delete(string& token,int c,int isary);

  // expression
 protected:
  G__value compile_expression(string& token);
  G__value compile_arglist(string& args,G__param* libp);
 private:
  int getstaticvalue(string& token);

  // blocks
  int compile_if(string& token,int c);
  int compile_switch(string& token,int c);
  int compile_for(string& token,int c);
  int compile_while(string& token,int c);
  int compile_do(string& token,int c);
  int compile_return(string& token,int c);
  int compile_throw(string& token,int c);
  int compile_catch(string& token,int c);
  int compile_try(string& token,int c);

  int compile_preprocessor(string& token,int c);

  // declaration
  int compile_declaration(G__TypeReader& type,string& token,int c);

 protected:
  struct G__var_array* allocatevariable(G__TypeReader& type
					,const string& name
					,int& ig15
					,deque<int>& arysize
					,deque<int>& typesize
                                        ,int isextrapointer);
 private:
  int readarraysize(deque<int>& arraysize);
  int readtypesize(string& token,deque<int>& typesize
			  ,int& isextrapointer);
  void setarraysize(G__TypeReader& type
		    ,struct G__var_array* var,int ig15
		    ,deque<int>& arysize
		    ,deque<int>& typesize
		    ,int isextrapointer) ;

  int init_reftype(string& token,struct G__var_array* var,int ig15,int c);

  int init_w_ctor(G__TypeReader& type
		  ,struct G__var_array* var,int ig15
		  ,string& token,int c);
  int init_w_defaultctor(G__TypeReader& type
			 ,struct G__var_array* var,int ig15
		         ,string& token,int c);

  int init_w_expr(G__TypeReader& type
		 ,struct G__var_array* var,int ig15
		 ,string& token,int c);

  int call_ctor(G__TypeReader& type,struct G__param *libp
		 ,struct G__var_array* var,int ig15,int num);


  //protected:
 public:
  G__value call_func(G__ClassInfo& cls
                     ,const string& fname,struct G__param *libp
                     ,int memfuncflag,int isarray=0
                    ,G__ClassInfo::MatchMode mode=G__ClassInfo::ConversionMatch
                     );

 public:
  int conversion(G__value& result,struct G__var_array* var,int ig15,int vartype,int paran);
  int baseconversion(G__value& result,struct G__var_array* var,int ig15,int vartype,int paran);
  int conversionopr(G__value& result,struct G__var_array* var,int ig15,int vartype,int paran);

 private:
  int read_initialization(G__TypeReader& type
			  ,struct G__var_array* var,int ig15
			  ,string& token,int c);
  int initscalar(G__TypeReader& type,struct G__var_array* var,int ig15
		  ,string& token,int c);
  int initstruct(G__TypeReader& type,struct G__var_array* var,int ig15
		  ,string& token,int c);
  int initscalarary(G__TypeReader& type,struct G__var_array* var,int ig15
		  ,string& token,int c);
  int initstructary(G__TypeReader& type,struct G__var_array* var,int ig15
		  ,string& token,int c);

  int Istypename(const string& name) ;
  int Isfunction(const string& name) ;

  long getstaticobject(const string& varname,struct G__ifunc_table* ifunc,int ifn,int noerror=0);

};

/***********************************************************************
 * static object
 ***********************************************************************/
extern G__blockscope *G__currentscope;

#endif

