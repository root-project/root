// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "Cintex/Cintex.h"
#include "CINTdefs.h"
#include "CINTVariableBuilder.h"
#include "CINTScopeBuilder.h"

#include "Api.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT {
namespace Cintex {

namespace {

bool IsDerivedFrom(Type type, const string& base_name)
{
   Type base_type = Type::ByName(base_name);
   if (!base_type)  {
      return false;
   }
   if ((type == base_type) || type.HasBase(base_type))  {
      return true;
   }
   return false;
}

} // unnamed namespace

//______________________________________________________________________________
//
//  Public Static Interface
//

//______________________________________________________________________________
void CINTVariableBuilder::Setup(const Member mbr)
{
   // Create a cint member variable for passed data member.
   //--
   //cerr << endl << endl << "Begin CINTVariableBuilder::Setup()" << endl << endl;
   //cerr << "mbr: " << mbr.Name(SCOPED) << endl;
   //
   //  Check if the data member has a valid type.
   //
   Type mbr_final_type = mbr.TypeOf().FinalType();
   if (!mbr_final_type) {
      if (mbr.IsTransient()) {
         if (Cintex::Debug()) {
            cout << "Cintex: Ignore transient member: " << mbr.Name(SCOPED) << " [No valid reflection class]" << endl;
         }
         //cerr << endl << endl << "End CINTVariableBuilder::Setup()" << endl << endl;
         return;
      }
      if (Cintex::Debug())  {
         cout << "Cintex: WARNING: Member: " << mbr.Name(SCOPED) << " [No valid reflection class]" << endl;
      }
   }
   //
   //  Get data member type, tagnum, typenum, and reftype.
   //
   char member_type = '\0';
   int member_reftype = 0;
   int member_tagnum = -1;
   int member_typenum = -1;
   {
      //
      //  Get data member type, and reftype.
      //
      Type pointed_at_type = mbr.TypeOf();
      for (; pointed_at_type.IsTypedef(); pointed_at_type = pointed_at_type.ToType()) {}
      for (; pointed_at_type.IsPointer(); pointed_at_type = pointed_at_type.ToType()) {
         ++member_reftype;
      }
      CintTypeDesc cint_type_desc = CintType(pointed_at_type);
      member_type = cint_type_desc.first;
      if (member_reftype)  {
         member_type = toupper(member_type);
         if (member_reftype == 1) {
            member_reftype = 0;
         }
      }
      //
      //  Get data member tagnum and typenum.
      //
      if (cint_type_desc.first == 'u')  {
         if (mbr.Properties().HasProperty("iotype")) {
            member_tagnum = CintTag(mbr.Properties().PropertyAsString("iotype"));
         }
         else {
            member_tagnum = CintTag(cint_type_desc.second);
         }
#if defined(_WIN32) && !defined(__CINT__)
         typedef __int64 longlong;
         typedef unsigned __int64 ulonglong;
#else // _WIN32 && !__CINT__
         typedef long long int longlong;
         typedef unsigned long long int ulonglong;
#endif // _WIN32 && !__CINT__
         if (pointed_at_type.TypeInfo() == typeid(longlong)) {
            G__loadlonglong(&member_tagnum, &member_typenum, G__LONGLONG);
         }
         else if (pointed_at_type.TypeInfo() == typeid(ulonglong)) {
            G__loadlonglong(&member_tagnum, &member_typenum, G__ULONGLONG);
         }
         else if (pointed_at_type.TypeInfo() == typeid(long double)) {
            G__loadlonglong(&member_tagnum, &member_typenum, G__LONGDOUBLE);
         }
      }
   }
   //
   //  Get data member storage duration.
   //
   int member_statictype = G__AUTO;
   if (mbr.IsStatic() || mbr.DeclaringScope().IsNamespace()) {
      member_statictype = G__LOCALSTATIC;
   }
   //
   //  Get data member access restriction.
   //
   int member_access = 0;
   if (mbr.IsPrivate()) {
      member_access = G__PRIVATE;
   }
   else if (mbr.IsProtected()) {
      member_access = G__PROTECTED;
   }
   else if (mbr.IsPublic()) {
      member_access = G__PUBLIC;
   }
   //
   //  Get i/o name and initializer expression.
   //
   string ioname;
   if (mbr.Properties().HasProperty("ioname")) {
      ioname = mbr.Properties().PropertyAsString("ioname");
   }
   else {
      ioname = mbr.Name();
   }
   ostringstream expr;
   if (mbr_final_type.IsArray()) {
      expr << ioname << "[" << mbr_final_type.ArrayLength() << "]=";
   }
   else {
      expr << ioname << "=";
   }
   //
   //  Get comment string.
   //
   string comment;
   if (mbr.Properties().HasProperty("comment")) {
      comment = mbr.Properties().PropertyAsString("comment");
      //cerr << endl << "##########" << endl << "mbr: " << mbr.Name(SCOPED) << "  comment: '" << comment << "'" << endl << "##########" << endl;
   }
   else {
      //cerr << endl << "##########" << endl << "mbr: " << mbr.Name(SCOPED) << "  comment: No comment property found." << endl << "##########" << endl;
   }
   //
   //  Add any necessary artificial comments.
   //
   const char* ref_t = "pool::Reference";
   const char* tok_t = "pool::Token";
   Type declaring_class = mbr.DeclaringScope();
   if (
      mbr.IsTransient() ||
      IsSTL(CintName(declaring_class)) ||
      IsDerivedFrom(declaring_class, ref_t) ||
      IsDerivedFrom(declaring_class, tok_t)
   ) {
      comment = "! " + comment;
   }
   else if (
      mbr_final_type.IsClass() &&
      (
         IsDerivedFrom(mbr_final_type, ref_t) ||
         IsDerivedFrom(mbr_final_type, tok_t)
      )
   ) {
      comment = "|| " + comment;
   }
   //
   //  Debug printout.
   //
   if (Cintex::Debug() > 2)  {
      cout
         << setw(24) << left << "Cintex: declareField>"
         << "  [" << char(member_type)
         << "," << right << setw(3) << mbr.Offset()
         << "," << right << setw(2) << member_reftype
         << "," << right << setw(3) << member_tagnum
         << "] "
         << (mbr.TypeOf().IsConst() ? "const " : "")
         << left << setw(7)
         << (G__AUTO == member_statictype ? "auto " : "static ")
         << left << setw(24) << ioname
         << " \"" << (char*)(!comment.empty() ? comment.c_str() : "(None)") << "\""
         << endl
         << setw(24) << left << "Cintex: declareField>"
         << "  Type:"
         << left << setw(24) << ("[" + mbr.Properties().HasProperty("iotype") ? mbr.Properties().PropertyAsString("iotype") : mbr_final_type.Name(SCOPED) + "]")
         << " DeclBy:" << mbr.DeclaringScope().Name(SCOPED)
         << endl;
   }
   //
   //  Call the cint dictionary interface
   //  to add the data member.
   //
   //cerr << "CINTVariableBuilder::Setup: " << mbr.Name(SCOPED) << "  comment: '" << comment.c_str() << "'" << endl;
   char* comment_ptr = (char*) ::operator new(comment.size() + 1); // FIXME: This creates a memory leak, the cint dictionary passes string constants from the dictionary implementation file, so there is no memory release done on these strings.
   strcpy(comment_ptr, comment.c_str());
   G__memvar_setup(
        (void*) mbr.Offset() // p
      , member_type // type
      , member_reftype // reftype
      , mbr.TypeOf().IsConst() // constvar
      , member_tagnum // tagnum
      , member_typenum // typenum
      , member_statictype // statictype
      , member_access // accessin
      , expr.str().c_str() // expr
      , 0 // definemacro
      , comment_ptr // comment
   );
   //cerr << endl << endl << "End CINTVariableBuilder::Setup()" << endl << endl;
}

//______________________________________________________________________________
//
//  Public Interface
//

//______________________________________________________________________________
CINTVariableBuilder::CINTVariableBuilder(const ROOT::Reflex::Member& m)
      : fVariable(m)
{
}

//______________________________________________________________________________
CINTVariableBuilder::~CINTVariableBuilder()
{
}

//______________________________________________________________________________
void CINTVariableBuilder::Setup()
{
   // Insure the 'scopes' for this variable are setup properly.

   CINTScopeBuilder::Setup(fVariable.TypeOf());
   Scope scope = fVariable.DeclaringScope();
   CINTScopeBuilder::Setup(scope);
   bool global = scope.IsTopScope();
   if (global) {
      G__resetplocal();
   }
   else {
      string sname = scope.Name(SCOPED);
      int stagnum = G__defined_tagname(sname.c_str(), 2);
      G__tag_memvar_setup(stagnum);
   }
   Setup(fVariable);
   if (global) {
      G__resetglobalenv();
   }
   else {
      G__tag_memvar_reset();
   }
   return;
}

} // namespace Cintex
} // namespace ROOT
