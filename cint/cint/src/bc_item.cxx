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
G__value G__blockscope_expr::getitem(const string& item_string)
{
   int i = 0;
   int c;
   const char *item = item_string.c_str();
   while ((c = item[i])) {
      switch (c) {
         case ':':
            if (':' == item[i+1]) {
               return scope_operator(item, i);
            }
            else {
               //error
            }
            break;

         case '.':
            return member_operator(item, i);
            break;

         case '-':
            if ('>' == item[i+1]) {
               return pointer_operator(item, i);
            }
            else {
               //error
            }
            break;

         case '[':
            return index_operator(item, i);
            break;

         case '(':
            return fcall_operator(item, i);
            break;

         default:
            break;
      }
      ++i;
   }
   G__object_id objid;
   return getobject(item,&objid);
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
   G__object_id objid;
   G__value obj = getobject(name,&objid);
   m_localscope.Init(obj.tagnum);
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
   G__object_id objid;
   G__value obj = getobject(name,&objid);
   m_isfixed = 0;
   G__TypeReader ty(obj);

   if (ty.Ispointer() && -1 != ty.Tagnum()) {
      //pointer->member
      //  G__getexpr(pointer)
      //  PUSHSTROS
      //  SETSTROS
      //  -> member
      //  POPSTROS

      m_localscope.Init(obj.tagnum);
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

      struct G__param* para = new G__param();
      para->paran = 0;
      obj = m_blockscope->call_func(m_localscope, "operator->", para, 0, 0
                                             , G__ClassInfo::ExactMatch);
      delete para;
      m_localscope.Init(obj.tagnum);
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
         m_pinst->LD_LVAR(objid.m_var, objid.m_ig15, arrayindex, 'p');
      }
      else if (objid.IsGlobal()) {
         m_pinst->LD_VAR(objid.m_var, objid.m_ig15, arrayindex, 'p');
      }
      else if (objid.IsMember()) {
         m_pinst->LD_MSTR(objid.m_var, objid.m_ig15, arrayindex, 'p');
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
   
   if (sindex.size() && !objid.Ispointer() && (objid.Tagnum() != -1)) {
      struct G__param* para = new G__param();
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
         para->paran = 1;
         m_pinst->SETMEMFUNCENV();
         para->para[0] = m_blockscope->compile_expression(sindex.front());
         sindex.pop_front();
         m_pinst->RECMEMFUNCENV();
         m_pinst->SWAP();
         m_pinst->PUSHSTROS();
         m_pinst->SETSTROS();
         m_localscope.Init(objid.Tagnum());
         G__value obj = m_blockscope->call_func(m_localscope, "operator[]", para, 0, 0, G__ClassInfo::ExactMatch);
         objid.SetVar(0, -1, G__object_id::VAR_NON);
         objid.SetIfunc(0, -1);
         objid.SetObj(obj);
         m_pinst->POPSTROS();
      }
      delete para;
   }
   /////////////////////////////////////////////////////////////////
   G__value result = G__null;
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
      m_pinst->LD_LVAR(objid->m_var, objid->m_ig15, arrayindex, 'p');
   }
   else if (objid->IsGlobal()) {
      m_pinst->LD_VAR(objid->m_var, objid->m_ig15, arrayindex, 'p');
   }
   else if (objid->IsMember()) {
      m_pinst->LD_MSTR(objid->m_var, objid->m_ig15, arrayindex, 'p');
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

   const char* pname = name.c_str();
   int hash, ig15;
   G__hash(name, hash, ig15);
   struct G__var_array *var;
   struct G__var_array *orig_var = m_blockscope->m_var;
   int tagnum = m_blockscope->m_var->tagnum;

   //1. block scope in function
   //  block -> enclosing block  var = G__blockscope::m_var
   //                              var->enclosing_scope
   //orig_var=m_blockscope->m_var;
   while (orig_var) {
      var = orig_var;
      while (var) {
         for (ig15 = 0;ig15 < var->allvar;ig15++) {
            if (hash == var->hash[ig15] && strcmp(pname, var->varnamebuf[ig15]) == 0) {
               if (id) id->SetVar(var, ig15, G__object_id::VAR_LOCAL);
               goto l_match;
            }
         }
         var = var->next;
      }
      orig_var = orig_var->enclosing_scope;
   }

   //2. class scope -> base class (or using namespace)
   //  tag -> base             tagnum=G__blockscope::m_ifunc->tagnum[m_iexist]
   //   |     using scope        basciass=G__struct.baseclass
   //int tagnum = m_blockscope->m_var->tagnum;
   if (-1 != tagnum) {
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

      // base class or using namespace
      int ib, btagnum;
      for (ib = 0;ib < G__struct.baseclass[tagnum]->basen;ib++) {
         btagnum = G__struct.baseclass[tagnum]->herit[ib]->basetagnum;
         var = G__struct.memvar[btagnum];
         while (var) {
            for (ig15 = 0;ig15 < var->allvar;ig15++) {
               if (hash == var->hash[ig15] && strcmp(pname, var->varnamebuf[ig15]) == 0) {
                  if (id) id->SetVar(var, ig15, G__object_id::VAR_MEMBER);
                  goto l_match;
               }
            }
            var = var->next;
         }
      }
   }

   //3. class/namespace scope -> enclosing scope -> global scope
   //  enclosing scope -> global   next_tagnum=G__struct.parent_tagnum[tagnum]
   if (-1 != tagnum) {
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

   // l_unmatch: //////////////////////////////////////////////////////
   if (id) {
      id->SetVar(0, -1, G__object_id::VAR_NON);
      id->SetIfunc(0, -1);
      id->SetObj(G__null);
   }
   return(G__null);
l_match:
   G__value result;
   result.type = var->type[ig15];
   result.tagnum = var->p_tagtable[ig15];
   result.typenum = var->p_typetable[ig15];
   result.obj.reftype.reftype = var->reftype[ig15];
   result.ref = 1; // dummy that shows there is valid object reference
   result.isconst = var->constvar[ig15];
   if (id) {
      //id->SetVar(var,ig15);
      id->SetIfunc(0, -1);
      id->SetObj(result);
   }
   return(result);
}

//______________________________________________________________________________
G__ClassInfo G__blockscope_expr::getscope(const string& name)
{
   int hash, ig15;
   const char* pname = name.c_str();
   G__hash(name, hash, ig15);
   for (int i = 0;i < G__struct.alltag;i++) {
      if (hash == G__struct.hash[i] && strcmp(G__struct.name[i], pname) == 0) {
         if (m_isfixed && m_localscope.Tagnum() != G__struct.parent_tagnum[i]) {
            continue;
         }
         if (-1 != G__struct.parent_tagnum[i]) {
            int tagnum = m_blockscope->GetTagnum();
            int j = i;
            while (-1 != j) {
               if (j == tagnum) {
                  G__ClassInfo scope(i);
                  return(scope);
               }
               j = G__struct.parent_tagnum[j];
            }
            continue;
         }
         G__ClassInfo scope(i);
         return(scope);
      }
   }
   return G__ClassInfo();
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
extern "C" G__value G__bc_getitem(char* item)
{
   G__blockscope_expr expr(G__currentscope);
   return(expr.getitem(string(item)));
}

