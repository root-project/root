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

#include "G__ci.h"
#include "common.h"

#include "bc_inst.h"
#include "bc_type.h"
#include "bc_reader.h"

#include <map>
#include <vector>
#include <deque>
#include <string>

namespace Cint {
   namespace Bytecode {

typedef int G__bc_pointer;
typedef int G__bc_pointer_addr;


////////////////////////////////////////////////////////////////////
// Virtual base class initialization
////////////////////////////////////////////////////////////////////
void G__bc_Baseclassctor_vbase(int tagnum);

////////////////////////////////////////////////////////////////////
// Generate instruction
////////////////////////////////////////////////////////////////////
void G__bc_VIRTUALADDSTROS(int tagnum
                                      ,struct G__inheritance* baseclas
                                      ,int basen) ;
void G__bc_cancel_VIRTUALADDSTROS() ;
void G__bc_REWINDSTACK(int n) ;
int G__bc_casejump(void* p,int val);

////////////////////////////////////////////////////////////////////
// G__bc_new_operator
////////////////////////////////////////////////////////////////////
G__value G__bc_new_operator(const char *expression) ;
void G__bc_delete_operator(const char *expression,int isarray) ;

////////////////////////////////////////////////////////////////////
// G__Isvalidassignment
////////////////////////////////////////////////////////////////////
int G__Isvalidassignment(G__TypeReader& ltype,G__TypeReader& rtype,G__value* rval);
int G__Isvalidassignment_val(G__value* ltype,int varparan,int lparan,int lvar_type,G__value* rtype);
int G__bc_conversion(G__value *result,const ::Reflex::Member &var
                                ,int var_type,int paran) ;

/***********************************************************************
 * G__jumptable
 ***********************************************************************/
class G__breaktable {
  std::vector<int> m_breaktable;
 public:
  void add(int originpc) { m_breaktable.push_back(originpc); }
  void resolve(G__bc_inst& inst,int destination); 
};

/***********************************************************************
 * G__casetable 
 ***********************************************************************/
class G__casetable {
  std::map<long,long> m_casetable;
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
  std::map<std::string,int> m_labeltable;
  std::map<std::string,int> m_gototable;
 public:
  void clear() { m_labeltable.clear(); m_gototable.clear(); }
  void addlabel(const std::string& label,int labelpc) {m_labeltable[label]=labelpc;}
  void addgoto(int originpc,const std::string& label) {m_gototable[label]=originpc;}
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
  ::Reflex::Scope  m_ifunc;
  ::Reflex::Member m_iexist;

  Reflex::Scope    m_scope;
  Reflex::Scope    store_p_local;

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
  // int fgettoken(std::string& token);

  // parsing 
  // top entry for compilation
  int compile(int openBrace=0);
  int compile_core(int openBrace=0);

  // called from G__bc_new_operator
  G__value compile_newopr(const std::string& expression);
  void compile_deleteopr(std::string& expression,int isarray);

  // Initialize virtual base class. Moved from G__functionscope
  void Baseclassctor_vbase(int tagnum);

  // access rule check
  int access(/* const */ G__MethodInfo& x) const;
  int access(/* const */ G__DataMemberInfo& x) const;

  int access(const Reflex::Scope& tagnum,long property) const;
  int isfriend(const Reflex::Scope& tagnum) const;

  G__bc_inst& GetInst() { return(m_bc_inst); }
  int GetTagnum() const { return(Cint::Internal::G__get_tagnum(m_ifunc.DeclaringScope())); }

 private:
  // operator
  int compile_space(std::string& token,int c);
  int compile_case(std::string& token,int c);
  int compile_default(std::string& token,int c);
  int compile_operator(std::string& token,int c);
  int compile_operator_PARENTHESIS(std::string& token,int c);
  int compile_operator_AND_ASTR(std::string& token,int c);
  int compile_operator_LESS(std::string& token,int c);
  int compile_operator_DIV(std::string& token,int c);
  int compile_bracket(std::string& token,int c);
  int compile_column(std::string& token,int c);
  int compile_semicolumn(std::string& token,int c);
  //int compile_quotation(std::string& token,int c);
  int compile_parenthesis(std::string& token,int c);
  int compile_brace(std::string& token,int c);
  int compile_new(std::string& token,int c);
  int compile_delete(std::string& token,int c,int isary);

  // expression
 protected:
  G__value compile_expression(std::string& token);
  G__value compile_arglist(std::string& args,G__param* libp);
 private:
  int getstaticvalue(std::string& token);

  // blocks
  int compile_if(std::string& token,int c);
  int compile_switch(std::string& token,int c);
  int compile_for(std::string& token,int c);
  int compile_while(std::string& token,int c);
  int compile_do(std::string& token,int c);
  int compile_return(std::string& token,int c);
  int compile_throw(std::string& token,int c);
  int compile_catch(std::string& token,int c);
  int compile_try(std::string& token,int c);

  int compile_preprocessor(std::string& token,int c);

  // declaration
  int compile_declaration(G__TypeReader& type,std::string& token,int c);

 protected:
    Reflex::Member allocatevariable(G__TypeReader& type
                                        ,const std::string& name
                                        ,std::deque<int>& arysize
                                        ,std::deque<int>& typesize
                                        ,int isextrapointer);
 private:
  int readarraysize(std::deque<int>& arraysize);
  int readtypesize(std::string& token,std::deque<int>& typesize
                          ,int& isextrapointer);

  int init_reftype(std::string& token,const Reflex::Member& var,int c);

  int init_w_ctor(G__TypeReader& type
                  ,const Reflex::Member& var
                  ,std::string& token,int c);
  int init_w_defaultctor(G__TypeReader& type
                         ,const Reflex::Member& var
                         ,std::string& token,int c);

  int init_w_expr(G__TypeReader& type
                 ,const Reflex::Member& var
                 ,std::string& token,int c);

  int call_ctor(G__TypeReader& type,struct G__param *libp
                 ,const Reflex::Member& var,int num);


  //protected:
 public:
  G__value call_func(G__ClassInfo& cls
                     ,const std::string& fname,struct G__param *libp
                     ,int memfuncflag,int isarray=0
                    ,G__ClassInfo::MatchMode mode=G__ClassInfo::ConversionMatch
                     );

 public:
  int conversion(G__value& result,const Reflex::Member& var,int vartype,int paran);
  int baseconversion(G__value& result,const Reflex::Member& var,int vartype,int paran);
  int conversionopr(G__value& result,const Reflex::Member& var,int vartype,int paran);

 private:
  int read_initialization(G__TypeReader& type
                          ,const Reflex::Member& var
                          ,std::string& token,int c);
  int initscalar(G__TypeReader& type,const Reflex::Member& var
                  ,std::string& token,int c);
  int initstruct(G__TypeReader& type,const Reflex::Member& var
                  ,std::string& token,int c);
  int initscalarary(G__TypeReader& type,const Reflex::Member& var
                  ,std::string& token,int c);
  int initstructary(G__TypeReader& type,const Reflex::Member& var
                  ,std::string& token,int c);

  int Istypename(const std::string& name) ;
  int Isfunction(const std::string& name) ;

  char* getstaticobject(const std::string& varname,const Reflex::Member& func,int noerror=0);

};

/***********************************************************************
 * static object
 ***********************************************************************/
extern G__blockscope *G__currentscope;

   } // namespace Bytecode
} // namespace Cint

#endif

