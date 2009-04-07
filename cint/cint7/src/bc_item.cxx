#if 0
/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_item.cxx
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

#include <cmath>
#include <deque>
#include <string>

using namespace std;

#include "bc_item.h"

namespace Cint {
namespace Bytecode {
using namespace ::Cint::Internal;

//______________________________________________________________________________
G__blockscope_expr::G__blockscope_expr(G__blockscope* blockscope)
{
   m_blockscope = blockscope;
   m_pinst = &blockscope->GetInst();
   m_isfixed = 0;
   m_isobject = 0;
   m_localscope.Init(-1);
}

//______________________________________________________________________________
G__value G__blockscope_expr::getitem(const string& item)
{
   G__value result;
   int i = 0;
   int c;
   while ((c = item[i])) {
      switch (c) {
         case ':':
            if (':' == item[i+1]) {
               result = scope_operator(item, i);
            }
            else {
               //error
            }
            break;

         case '.':
            result = member_operator(item, i);
            break;

         case '-':
            if ('>' == item[i+1]) {
               result = pointer_operator(item, i);
            }
            else {
               //error
            }
            break;

         case '[':
            result = index_operator(item, i);
            break;

         case '(':
            result = fcall_operator(item, i);
            break;

         case 0:
            result = getobject(item);
            break;

         default:
            break;
      }
      ++i;
   }
   return result;
}

//______________________________________________________________________________
G__value G__blockscope_expr::scope_operator(const string& item, int& i)
{
   //class_name::member
   //::member
   if (i == 0) {
      //::member
      m_isfixed = 1;
      m_localscope.Init();
      return(getitem(item, i + 2));
   }
   else {
      string scopename = item.substr(0, i);
      G__ClassInfo scope = getscope(scopename);
      if (scope.IsValid()) {
         m_isfixed = 1;
         m_localscope.Init(scope.Tagnum());
         return(getitem(item, i + 2));
      }
      // error
      G__fprinterr(G__serr, "Error: undefined scope name '%s'", scopename.c_str());
      G__genericerror((char*)NULL);
   }
   return(G__null);
}

//______________________________________________________________________________
G__value G__blockscope_expr::member_operator(const string& item, int& i)
{
   //object.member
   //  G__getexpr(object)
   //  PUSHSTROS
   //  SETSTROS
   //  -> member
   //  POPSTROS

   string name = item.substr(0, i);
   G__value obj = getobject(name);
   m_localscope.Init(G__get_tagnum(G__value_typenum(obj)));
   m_isobject = 1;
   m_isfixed = 0;

   m_pinst->PUSHSTROS();
   m_pinst->SETSTROS();

   G__value result = getitem(item, i + 1);

   m_pinst->POPSTROS();

   m_localscope.Init(-1);
   m_isobject = 0;

   return(result);
}

//______________________________________________________________________________
G__value G__blockscope_expr::pointer_operator(const string& item, int& i)
{
   //pointer->member
   //  G__getexpr(pointer)
   //  PUSHSTROS
   //  SETSTROS
   //  -> member
   //  POPSTROS
   //object->member      (object.operator->())->member
   //  G__getexpr(object)
   //  PUSHSTROS
   //  SETSTROS
   //  operator->
   //  PUSHSTROS
   //  SETSTROS
   //  -> member
   //  POPSTROS
   //  POPSTROS

   string name = item.substr(0, i);
   G__value obj = getobject(name);
   m_isfixed = 0;
   G__TypeReader ty(obj);

   if (ty.Ispointer() && -1 != ty.Tagnum()) {
      //pointer->member
      //  G__getexpr(pointer)
      //  PUSHSTROS
      //  SETSTROS
      //  -> member
      //  POPSTROS

      m_localscope.Init(G__get_tagnum(G__value_typenum(obj)));
      m_isobject = 1;

      m_pinst->PUSHSTROS();
      m_pinst->SETSTROS();

      G__value result = getitem(item, i + 2);

      m_pinst->POPSTROS();

      m_localscope.Init(-1);
      m_isobject = 0;
      return(result);
   }
   else if (ty.Ispointer() == 0 && -1 != ty.Tagnum()) {
      //object->member      (object.operator->())->member
      //  G__getexpr(object)
      //  PUSHSTROS
      //  SETSTROS
      //  operator->
      //  PUSHSTROS
      //  SETSTROS
      //  -> member
      //  POPSTROS
      //  POPSTROS

      m_localscope.Init(ty.Tagnum());
      m_isobject = 1;

      m_pinst->PUSHSTROS();
      m_pinst->SETSTROS();

      struct G__param para;
      para.paran = 0;
      G__value obj = m_blockscope->call_func(m_localscope, "operator->", &para, 0, 0
                                             , G__ClassInfo::ExactMatch);
      m_localscope.Init(G__get_tagnum(G__value_typenum(obj)));
      m_isobject = 1;
      m_isfixed = 0;

      m_pinst->PUSHSTROS();
      m_pinst->SETSTROS();

      G__value result = getitem(item, i + 1);

      m_pinst->POPSTROS();
      m_pinst->POPSTROS();

      m_localscope.Init(-1);
      m_isobject = 0;
      return(result);
   }
   else {
      // error
   }
   return G__null;
}

//______________________________________________________________________________
G__value G__blockscope_expr::index_operator(const string& item, int& i)
{
   string name = item.substr(0, i);
   G__object_id objid;
   searchobject(name, &objid);
   int arraydim = objid.ArrayDim();
   m_isfixed = 0;
   //o   varname [2][3]
   //o   varname [2][3](expr_list)
   //o   varname [2][3].member
   //o   varname [2][3]->member
   //             ^--->^
   deque<string> sindex;
   int c = readarrayindex(item, i, sindex);
   /////////////////////////////////////////////////////////////////
   // array
   int arrayindex = bc_min(arraydim, (int) sindex.size());
   if (arrayindex) {
      //array[expr][expr][expr]
      //   G__getexpr(expr)
      //   G__getexpr(expr)
      //   G__getexpr(expr)
      //   LD_VAR pointer index=3
      int j;
      m_pinst->SETMEMFUNCENV();
      for (j = 0; j < arrayindex; ++j) {
         m_blockscope->compile_expression(sindex.front());
         sindex.pop_front();
      }
      arraydim -= arrayindex;
      m_pinst->RECMEMFUNCENV();
      if (objid.IsLocal()) {
         m_pinst->LD_LVAR(objid.m_var, arrayindex, 'p');
      }
      else if (objid.IsGlobal()) {
         m_pinst->LD_VAR(objid.m_var, arrayindex, 'p');
      }
      else if (objid.IsMember()) {
         m_pinst->LD_MSTR(objid.m_var, arrayindex, 'p');
      }
   }
   /////////////////////////////////////////////////////////////////
   // array as pointer
   //if(arraydim && sindex.size()) {
   // never occur
   //}
   /////////////////////////////////////////////////////////////////
   // pointer
   while (sindex.size() && objid.Ispointer()) {
      //pointer[expr]
      //   object is already fetched by LD_VAR
      //   SETMEMFUNCENV
      //   G__getexpr(expr)
      //   RECMEMFUNCENV
      //
      m_pinst->SETMEMFUNCENV();
      m_blockscope->compile_expression(sindex.front());
      sindex.pop_front();
      m_pinst->RECMEMFUNCENV();
      //int size = objid.objsize();
      m_pinst->OP2('+'); // size is automatically taken into account
      m_pinst->TOVALUE(&objid.m_obj);
      objid.decplevel();
   }
   /////////////////////////////////////////////////////////////////
   // class object + operator[] overloading
   while (sindex.size() && !objid.Ispointer() && (objid.Tagnum() != -1)) {
      //object[expr]
      //   object is already fetched by LD_VAR
      //   SETMEMFUNCENV
      //   G__getexpr(expr)
      //   RECMEMFUNCENV
      //   SWAP
      //   PUSHSTROS
      //   SETSTROS
      //   LD_FUNC operator[] paran=1
      //   POPSTROS
      struct G__param para;
      para.paran = 1;
      m_pinst->SETMEMFUNCENV();
      para.para[0] = m_blockscope->compile_expression(sindex.front());
      sindex.pop_front();
      m_pinst->RECMEMFUNCENV();
      m_pinst->SWAP();
      m_pinst->PUSHSTROS();
      m_pinst->SETSTROS();
      m_localscope.Init(objid.Tagnum());
      G__value obj = m_blockscope->call_func(m_localscope, "operator[]", &para, 0, 0, G__ClassInfo::ExactMatch);
      objid.SetVar(Reflex::Dummy::Member(), G__object_id::VAR_NON);
      objid.SetIfunc(Reflex::Dummy::Member());
      objid.SetObj(obj);
      m_pinst->POPSTROS();
   }
   /////////////////////////////////////////////////////////////////
   G__value result;
   switch (c) {
      case 0:   // ary[1][2]
         break;
      case '.': // ary[1][2].member
         result = member_operator(item, i);
         break;
      case '-': // ary[1][2]->member
         if ('>' == item[i+1]) {
            result = pointer_operator(item, i);
         }
         else {
            //error
         }
         break;
      case '(': // ary[1][2](expr_list)
         result = fcall_operator(item, i);
         break;
   }
   return result;
}

//______________________________________________________________________________
G__value G__blockscope_expr::fcall_operator(const string& /* item */, int& /* i */)
{
   //(type)expr
   //   G__getexpr(expr)
   //   CAST type
   //(expr)
   //   G__getexpr(expr)
   //object(expr,expr)
   //   G__getexpr(expr)
   //   G__getexpr(expr)
   //   G__getexpr(object)
   //   PUSHSTROS
   //   SETSTROS
   //   LD_FUNC operator() paran
   //   POPSTROS
   //This happens the last, since function overloading makes it complicated
   //function(expr,expr)
   //   G__getexpr(expr)
   //   G__getexpr(expr)
   //   LD_FUNC function paran
   return G__null;
}

//______________________________________________________________________________
G__value G__blockscope_expr::getobject(const string& name, G__object_id* objid)
{
   G__value result = searchobject(name, objid);
   int arrayindex = 0;
   if (objid->IsLocal()) {
      m_pinst->LD_LVAR(objid->m_var, arrayindex, 'p');
   }
   else if (objid->IsGlobal()) {
      m_pinst->LD_VAR(objid->m_var, arrayindex, 'p');
   }
   else if (objid->IsMember()) {
      m_pinst->LD_MSTR(objid->m_var, arrayindex, 'p');
   }
   return(result);
}

//______________________________________________________________________________
G__value G__blockscope_expr::searchobject(const string& name, G__object_id* id)
{
   //block -> enclosing block  var = G__blockscope::m_var
   //                            var->enclosing_scope
   //tag -> base               tagnum=G__blockscope::m_ifunc->tagnum[m_iexist]
   // |     using scope          basciass=G__struct.baseclass
   //enclosing scope -> global   next_tagnum=G__struct.parent_tagnum[tagnum]

   Reflex::Member var;
   Reflex::Scope orig_var = m_blockscope->m_scope;

   //1. block scope in function
   //  block -> enclosing block  var = G__blockscope::m_var
   //                              var->enclosing_scope
   //orig_var=m_blockscope->m_var;
   while (1) {
      for (Reflex::Member_Iterator ig15 = orig_var.DataMember_Begin();ig15 != orig_var.DataMember_End();++ig15)
         if (ig15->Name() == name) {
            var = *ig15;
            if (id) id->SetVar(*ig15, G__object_id::VAR_LOCAL);
            goto l_match;
         }
      if (orig_var.IsTopScope()) break;
      orig_var = orig_var.DeclaringScope();
   };

   //2. class scope -> base class (or using namespace)
   //  tag -> base             tagnum=G__blockscope::m_ifunc->tagnum[m_iexist]
   //   |     using scope        basciass=G__struct.baseclass
   //int tagnum = m_blockscope->m_var->tagnum;
   if (m_blockscope->m_scope) {
      for (Reflex::Member_Iterator ig15 = m_blockscope->m_scope.DataMember_Begin(); ig15 != m_blockscope->m_scope.DataMember_End(); ++ig15) {
         if (ig15->Name() == name) {
            var = *ig15;
            if (id) id->SetVar(*ig15, G__object_id::VAR_MEMBER);
            goto l_match;
         }
      }

      // base class or using namespace
      for (int ib = 0;ib < G__struct.baseclass[G__get_tagnum(m_blockscope->m_scope)]->vec.size();ib++) {
         Reflex::Scope btagnum = G__Dict::GetDict().GetScope(G__struct.baseclass[G__get_tagnum(m_blockscope->m_scope)]->vec[ib].basetagnum);
         for (Reflex::Member_Iterator ig15 = btagnum.DataMember_Begin(); ig15 != btagnum.DataMember_End(); ++ig15) {
            if (ig15->Name() == name) {
               var = *ig15;
               if (id) id->SetVar(*ig15, G__object_id::VAR_MEMBER);
               goto l_match;
            }
         }
      }
   }
#if 0 // done by m_scope
   //3. class/namespace scope -> enclosing scope -> global scope
   //  enclosing scope -> global   next_tagnum=G__struct.parent_tagnum[tagnum]
   if (m_scope) {
      tagnum = G__struct.parent_tagnum[tagnum];
      while (-1 != tagnum) {
         var = G__struct.memvar[tagnum];
         while (var) {
            for (ig15 = 0;ig15 < var->allvar;ig15++) {
               if (hash == var->hash[ig15] && strcmp(pname, var->varnamebuf[ig15]) == 0) {
                  if (id) id->SetVar(var, ig15, G__object_id::VAR_MEMBER);
                  goto l_match;
               }
            }
            var = var->next;
         }
         tagnum = G__struct.parent_tagnum[tagnum];
      }
   }
#endif

#if 0 // done by m_scope
   // global scope
   var = &G__global;
   while (var) {
      for (ig15 = 0;ig15 < var->allvar;ig15++) {
         if (hash == var->hash[ig15] && strcmp(pname, var->varnamebuf[ig15]) == 0) {
            if (id) id->SetVar(var, ig15, G__object_id::VAR_GLOBAL);
            goto l_match;
         }
      }
      var = var->next;
   }
#endif

   // l_unmatch: //////////////////////////////////////////////////////
   if (id) {
      id->SetVar(Reflex::Dummy::Member(), G__object_id::VAR_NON);
      id->SetIfunc(Reflex::Dummy::Member());
      id->SetObj(G__null);
   }
   return(G__null);
l_match:
   G__value result;
   G__value_typenum(result) = var.TypeOf();
   result.ref = 1; // dummy that shows there is valid object reference
   if (id) {
      //id->SetVar(var,ig15);
      id->SetIfunc(Reflex::Dummy::Member());
      id->SetObj(result);
   }
   return(result);
}

//______________________________________________________________________________
G__ClassInfo G__blockscope_expr::getscope(const string& name)
{
   int tagnum_to_lookup_in = m_blockscope->GetTagnum();
   if (m_isfixed)
      tagnum_to_lookup_in = m_localscope.Tagnum();
   Reflex::Scope scope_to_lookup_in = G__Dict::GetDict().GetScope(tagnum_to_lookup_in);
   Reflex::Scope scope_found = scope_to_lookup_in.LookupScope(name);
   return G__ClassInfo(G__get_tagnum(scope_found));
}

//______________________________________________________________________________
int G__blockscope_expr::readarrayindex(const string& expr, int& i, deque<string>& sindex)
{
   //o   varname [2][3]
   //o   varname [2][3](expr_list)
   //o   varname [2][3].member
   //o   varname [2][3]->member
   //x v[varname [2][3]]
   //x f(varname [2][3])
   //x f(varname [2][3],expr_list)
   //             ^    ^
   int c;
   G__srcreader<G__sstream> stringreader;
   stringreader.Init(expr.c_str());
   stringreader.setspos(i);
   string indexexpr;

   do {
      c = stringreader.fgetstream_(indexexpr, "]" , 1); // "]"

      if (expr == "") {
         //error
      }
      sindex.push_back(indexexpr);

      c = stringreader.fgetstream_(indexexpr, "[]()=;,.-+*/%<>" , 0);
   }
   while (c == '[');

   i = stringreader.getpos();
   return(c); // c== '=' ';' ',' ')'
}

//______________________________________________________________________________
G__value G__bc_getitem(char* item)
{
   G__blockscope_expr expr(G__currentscope);
   return(expr.getitem(string(item)));
}

} // namespace Bytecode
} // namespace Cint
#endif // 0
