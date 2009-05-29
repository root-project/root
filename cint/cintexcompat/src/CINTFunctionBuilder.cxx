// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "CINTFunctionBuilder.h"

#include "CINTdefs.h"
#include "CINTFunctional.h"
#include "CINTScopeBuilder.h"
#include "CINTTypedefBuilder.h"
#include "StubContext.h"

#include "Reflex/Member.h"
#include "Reflex/Reflex.h"
#include "Reflex/Scope.h"
#include "Reflex/Tools.h"

#include "Api.h"

#include <cctype>
#include <string>

using namespace ROOT::Reflex;
using namespace std;

// from cint/src/common.h
#define G__CONSTFUNC 8

namespace {

std::string CintSignature(const ROOT::Reflex::Member& func) {
   // Argument signature is:
   // <type-char> <'object type name'> <typedef-name> <indirection> <default-value> <argument-Name>
   // i - 'Int_t' 0 0 option
   string signature;
   Type ft = func.TypeOf();
   for (size_t p = 0 ; p < ft.FunctionParameterSize(); ++p) {
      Type pt = ft.FunctionParameterAt(p);
      string arg_sig;
      Type indir_type = pt;
      int indir_cnt = 0;
      for (; indir_type.IsTypedef(); indir_type = indir_type.ToType()) {}
      for (; indir_type.IsPointer(); indir_type = indir_type.ToType()) {
         ++indir_cnt;
      }
      ROOT::Cintex::CintTypeDesc ctype = ROOT::Cintex::CintType(indir_type);
      if (indir_cnt == 0) {
         arg_sig += ctype.first;
      }
      else {
         arg_sig += toupper(ctype.first);  // if pointer: 'f' -> 'F' etc.
      }
      arg_sig += " ";
      //---Fill the true-type and eventually the typedef
      if (ctype.second == "-") {
         arg_sig += "-";
         if (pt.IsTypedef()) {
            arg_sig += " '" + ROOT::Cintex::CintName(pt.Name(SCOPED)) + "' ";
         }
         else {
            arg_sig += " - ";
         }
      }
      else {
         G__TypedefInfo tdef(ctype.second.c_str());
         int tagnum = G__defined_tagname(ctype.second.c_str(), 2);
         if (tdef.IsValid()) {
            arg_sig += "'" + string(tdef.TrueName()) + "'";
         }
         else if (tagnum != -1) {
            arg_sig += "'" + string(G__fulltagname(tagnum, 1)) + "'";
         }
         else {
            arg_sig += "'" + ctype.second + "'";  // object type name
         }
         if (pt.IsTypedef() || tdef.IsValid()) {
            arg_sig += " '" + ROOT::Cintex::CintName(pt.Name(SCOPED)) + "' ";
         }
         else {
            arg_sig += " - ";
         }
      }
      // Assign indirection. First indirection already taken into account by uppercasing type
      if (!indir_cnt || (indir_cnt == 1)) {
         if (pt.IsReference() && pt.IsConst()) {
            arg_sig += "11";
         }
         else if (pt.IsReference()) {
            arg_sig += "1";
         }
         else if (pt.IsConst()) {
            arg_sig += "10";
         }
         else {
            arg_sig += "0";
         }
      }
      else {
         arg_sig += char('0' + indir_cnt); // convert 2 -> '2', 3 ->'3' etc.
      }
      arg_sig += " ";
      // Default value
      if (func.FunctionParameterDefaultAt(p) != "") {
         arg_sig += "'" + func.FunctionParameterDefaultAt(p) + "'";
      }
      else {
         arg_sig += "-";
      }
      arg_sig += " ";
      // Parameter Name
      if (func.FunctionParameterNameAt(p) != "") {
         arg_sig += func.FunctionParameterNameAt(p);
      }
      else {
         arg_sig += "-";
      }
      signature += arg_sig;
      if (p < (ft.FunctionParameterSize() - 1)) {
         signature += " ";
      }
   }
   return signature;
}

} // unnamed namespace

namespace ROOT {
namespace Cintex {

//______________________________________________________________________________
//
//  Static Interface
//

//______________________________________________________________________________
void CINTFunctionBuilder::Setup(ROOT::Reflex::Member function)
{
   // Insure that the Cint specific data is set for the given function.

   Type rt = function.TypeOf().ReturnType();
   int ret_typedeft = -1;
   if (rt.IsTypedef()) {
      ret_typedeft = CINTTypedefBuilder::Setup(rt);
      for (; rt.IsTypedef(); rt = rt.ToType()) {}
   }
   CintTypeDesc ret_desc = CintType(rt);
   char ret_type = rt.IsPointer() ? (ret_desc.first - ('a' - 'A')) : ret_desc.first;
   int ret_tag = CintTag(ret_desc.second);
   string funcname = function.Name();
   if (function.IsOperator()) {
      if ((funcname[8] == ' ') && !isalpha(funcname[9])) {
         funcname = "operator" + funcname.substr(9);
      }
   }
   StubContext_t* stub_context = new StubContext_t(function);
   G__InterfaceMethod stub;
   if (function.IsConstructor()) {
      stub = Allocate_stub_function(stub_context, &Constructor_stub_with_context);
      int tagnum = CintTag(function.DeclaringScope().Name(SCOPED));
      funcname = G__ClassInfo(tagnum).Name();
      ret_tag = tagnum;
   }
   else if (function.IsDestructor()) {
      stub = Allocate_stub_function(stub_context, &Destructor_stub_with_context);
      int tagnum = CintTag(function.DeclaringScope().Name(SCOPED));
      funcname = "~";
      funcname += G__ClassInfo(tagnum).Name();
   }
   else {
      stub = Allocate_stub_function(stub_context, &Method_stub_with_context);
   }
   int hash = 0;
   int tmp = 0;
   G__hash(funcname, hash, tmp);
   int reference = function.TypeOf().ReturnType().IsReference();
   int nparam = function.TypeOf().FunctionParameterSize();
   int memory_type = 1; // flag, ansi-style function parameters, not varadic
   if (function.IsStatic()) {
      memory_type |= 0x02; // flag static function
   }
   // FIXME: We are missing the varadic flag, and the explicit flag in memory_type.
   int access = G__PUBLIC;
   if (function.IsPrivate()) {
      access = G__PRIVATE;
   }
   else if (function.IsProtected()) {
      access = G__PROTECTED;
   }
   else if (function.IsPublic()) {
      access = G__PUBLIC;
   }
   int isconst = 0;
   if (function.TypeOf().IsConst()) {
      isconst = G__CONSTFUNC;
   }
   string signature = CintSignature(function);
   int isvirtual = 0;
   if (function.IsVirtual()) {
      isvirtual = 1;
   }
   // FIXME: We are missing the pure virtual flag in isvirtual.
   G__usermemfunc_setup(
      const_cast<char*>(funcname.c_str()), // funcname, function name
      hash,         // hash, function name hash value
      (int (*)(void)) stub, // funcp, method stub function
      ret_type,     // type, return type
      ret_tag,      // tagnum, return type
      ret_typedeft, // typenum, return type
      reference,    // reftype, return type
      nparam,       // para_nu, number of parameters
      memory_type,  // ansi, ansi flag
      access,       // accessin, access type
      isconst,      // isconst, return type
      const_cast<char*>(signature.c_str()), // paras, signature
      0,            // comment, comment line
      0,            // truep2f
      isvirtual,    // isvirtual
      stub_context // userparam
   );
}

//______________________________________________________________________________
//
//  Non-Static Interface
//

//______________________________________________________________________________
CINTFunctionBuilder::CINTFunctionBuilder(ROOT::Reflex::Member mbr)
      : fFunction(mbr)
{
}

//______________________________________________________________________________
CINTFunctionBuilder::~CINTFunctionBuilder()
{
}

//______________________________________________________________________________
void CINTFunctionBuilder::Setup()
{
   // Insure that the 'scopes' for the function are properly setup.

   Scope scope = fFunction.DeclaringScope();
   bool is_global = scope.IsTopScope();
   CINTScopeBuilder::Setup(fFunction.TypeOf());
   if (is_global) {
      G__lastifuncposition();
   }
   else {
      CINTScopeBuilder::Setup(scope);
      string sname = scope.Name(SCOPED);
      int ns_tag = G__search_tagname(sname.c_str(), 'n');
      G__tag_memfunc_setup(ns_tag);
   }
   Setup(fFunction);
   if (is_global) {
      G__resetifuncposition();
   }
   else {
      G__tag_memfunc_reset();
   }
   return;
}

} // namespace Cintex
} // namespace ROOT
