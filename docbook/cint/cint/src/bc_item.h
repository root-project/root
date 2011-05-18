/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_item.h
 ************************************************************************
 * Description:
 *  item expression compiler
 *   object
 *   function
 *   object.member
 *   ::member
 *   object.member
 *   pointer->member
 *   object->member      (object.operator->())->member
 *   pointer[expr]
 *   array[expr][expr][expr]
 *   object[expr]
 *   (type)expr
 *   (expr)
 *   object(expr,expr)
 *   function(expr,expr)
 ************************************************************************
 * Copyright(c) 2004~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef BC_ITEM_H
#define BC_ITEM_H

//extern "C" {
#ifdef __CINT__
#include "../G__ci.h"
#else
#include "common.h"
#endif
//}

#include "bc_parse.h"

////////////////////////////////////////////////////////////////////
// G__object_id
////////////////////////////////////////////////////////////////////
struct G__object_id : public G__TypeReader {
 public:
  enum VAR_ID { VAR_NON, VAR_GLOBAL, VAR_LOCAL, VAR_MEMBER };

  int ArrayDim() const { return 0; }
  void SetVar(struct G__var_array* var,int ig15,VAR_ID id) 
  { m_var=var; m_ig15=ig15; m_id=id; }
  void SetIfunc(struct G__ifunc_table* ifunc,int ifn) 
    { m_ifunc=ifunc; m_ifn=ifn; }
  void SetObj(G__value& obj) { m_obj=obj; G__TypeReader::Init(obj); }

  int IsLocal() const { return((m_id==VAR_LOCAL)?1:0); }
  int IsGlobal() const { return((m_id==VAR_GLOBAL)?1:0); }
  int IsMember() const { return((m_id==VAR_MEMBER)?1:0); }

 private:
  VAR_ID  m_id;
 public:
  struct G__var_array *m_var;
  int m_ig15;
  struct G__ifunc_table *m_ifunc;
  int m_ifn;
  //G__TypeReader m_type;
  G__value m_obj;
};


////////////////////////////////////////////////////////////////////
// G__blockscope_expr
////////////////////////////////////////////////////////////////////
class G__blockscope_expr {
 public:
  G__blockscope_expr(G__blockscope* blockscope);
  ~G__blockscope_expr() {}

  G__value getitem(const string& item);  

 private:
  G__value getitem(const string& item,int i) 
    { return(getitem(item.substr(i))); }

  G__value scope_operator(const string& item,int& i);
  G__value member_operator(const string& item,int& i);
  G__value pointer_operator(const string& item,int& i);
  G__value index_operator(const string& item,int& i);
  G__value fcall_operator(const string& item,int& i);

  int readarrayindex(const string& expr,int& i,deque<string>& sindex) ;

 private:
  G__value getobject(const string& name,G__object_id* id=0);
  G__value searchobject(const string& name,G__object_id* id=0);
  G__ClassInfo getscope(const string& name);
  G__TypeInfo gettype(const string& name);
  G__MethodInfo getfunction(const string& name);

 private:
  G__blockscope *m_blockscope;
  G__bc_inst* m_pinst;
  int m_isfixed;
  int m_isobject;
  G__ClassInfo m_localscope;
};


////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////


#endif
