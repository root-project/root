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
#include "CINTCommentBuffer.h"

#include "Api.h"

#include <iomanip>
#include <sstream>

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {
  

   CINTVariableBuilder::CINTVariableBuilder(const ROOT::Reflex::Member& m)
      : fVariable(m) { 
      // Default constructor.
   }
   
   CINTVariableBuilder::~CINTVariableBuilder() {
      // Destructor.
   }

   void CINTVariableBuilder::Setup() {
      // Setup variable info.
      CINTScopeBuilder::Setup(fVariable.TypeOf());

      Scope scope = fVariable.DeclaringScope();
      CINTScopeBuilder::Setup(scope);
    
      bool global = scope.IsTopScope();

      if ( global ) {
         ::G__resetplocal();
      }    
      else {
         string sname = scope.Name(SCOPED);
         int stagnum = ::G__defined_tagname(sname.c_str(), 2);
         ::G__tag_memvar_setup(stagnum);      
      }

      Setup(fVariable);
    
      if ( global ) {
         ::G__resetglobalenv();
      }
      else {
         ::G__tag_memvar_reset();
      }
      return;
   }

   void CINTVariableBuilder::Setup(const Member& dm ) {
      // Setup variable info.
      char* comment = NULL;
    
      const char* ref_t = "pool::Reference";
      const char* tok_t = "pool::Token";

      Type fClass = Type::ByName(dm.DeclaringScope().Name(SCOPED));
      string fName(CintName(fClass));

      std::string cm = dm.Properties().HasProperty("comment") ? 
         dm.Properties().PropertyAsString("comment") : std::string("");

      Type dmType = dm.TypeOf();
      if (dm.Properties().HasProperty("iotype")) {
         dmType = Reflex::Type::ByName(dm.Properties().PropertyAsString("iotype"));
      }
      while ( dmType.IsTypedef() ) dmType = dmType.ToType();
      if ( !dmType && dm.IsTransient() )  {
         // Before giving up, let's ask CINT:
         int tagnum = G__defined_tagname(dmType.Name().c_str(), 2);
         
         if (tagnum < 0) {
            if( Cintex::Debug() ) cout << "Cintex: Ignore transient member: " 
               << dm.Name(SCOPED) << " [No valid reflection class]"
               << " " << dmType.Name() << endl;
            return;
         }
      }
      else if ( !dmType )  {
         if( Cintex::Debug() > 0 )  {
            cout << "Cintex: WARNING: Member: " 
                 << dm.Name(SCOPED) << " [No valid reflection class]" << endl;
         }
         //throw std::runtime_error("Member: "+fName+"::"+dm.Name()+" [No valid reflection class]");
      }
      //--- Add the necessary artificial comments ------ 
      if ( dm.IsTransient() ||
           IsSTL(fName) || 
           IsTypeOf(fClass, ref_t) || 
           IsTypeOf(fClass, tok_t) )  {
         char* com = new char[cm.length()+4];
         ::sprintf(com,"! %s",cm.c_str());
         comment = com;
         CommentBuffer::Instance().Add(comment);
      }
      else if ( (dmType.IsClass() || dmType.IsStruct()) && 
                (IsTypeOf(dmType,ref_t) || IsTypeOf(dmType,tok_t)) )  {
         char* com = new char[cm.length()+4];
         ::sprintf(com,"|| %s",cm.c_str());
         comment = com;
         CommentBuffer::Instance().Add(comment);
      }
      else if ( !cm.empty() )  {
         char* com = new char[cm.length()+4];
         ::strcpy(com, cm.c_str());
         comment = com;
         CommentBuffer::Instance().Add(comment);
      }

      Indirection  indir = IndirectionGet(dmType);
      CintTypeDesc type = CintType(indir.second);
      string dname = dm.Properties().HasProperty("ioname") ? dm.Properties().PropertyAsString("ioname") : dm.Name();
      ostringstream ost;
      if ( dmType.IsArray() ) ost << dname << "[" << dmType.ArrayLength() << "]=";
      else                    ost << dname << "=";
      string expr = ost.str();
      int member_type     = type.first;
      int member_indir    = 0;
      int member_tagnum   = -1;
      int member_typnum   = -1;
      int member_isstatic = G__AUTO;
      if (dm.IsStatic() || dm.DeclaringScope().IsNamespace())
         member_isstatic = G__LOCALSTATIC;
      switch(indir.first)  {
      case 0: 
         break;
      case 1:
         member_type -= 'a'-'A';            // if pointer: 'f' -> 'F' etc.
         break;
      default:
         member_type -= 'a'-'A';            // if pointer: 'f' -> 'F' etc.
         member_indir = indir.first;
         break;
      }

      if ( type.first == 'u' )  {
         // Large integer definition depends of the platform
#if defined(_WIN32) && !defined(__CINT__)
         typedef __int64 longlong;
         typedef unsigned __int64 ulonglong;
#else
         typedef long long int longlong; /* */
         typedef unsigned long long int /**/ ulonglong;
#endif

         //dependencies.push_back(indir.second);
         member_tagnum = dm.Properties().HasProperty("iotype") ? CintTag(dm.Properties().PropertyAsString("iotype")) : CintTag(type.second);
         if ( typeid(longlong) == indir.second.TypeInfo() )
            ::G__loadlonglong(&member_tagnum, &member_typnum, G__LONGLONG);
         else if ( typeid(ulonglong) == indir.second.TypeInfo() )
            ::G__loadlonglong(&member_tagnum, &member_typnum, G__ULONGLONG);
         else if ( typeid(long double) == indir.second.TypeInfo() )
            ::G__loadlonglong(&member_tagnum, &member_typnum, G__LONGDOUBLE);
      }

      int member_access = 0;
      if ( dm.IsPrivate() )        member_access = G__PRIVATE;
      else if ( dm.IsProtected() ) member_access = G__PROTECTED;
      else if ( dm.IsPublic() )    member_access = G__PUBLIC;

      if ( Cintex::Debug() > 2 )  {
         std::cout 
            << std::setw(24) << std::left << "Cintex: declareField>"
            << "  [" << char(member_type) 
            << "," << std::right << std::setw(3) << dm.Offset()
            << "," << std::right << std::setw(2) << member_indir 
            << "," << std::right << std::setw(3) << member_tagnum
            << "] " 
            << (dmType.IsConst() ? "const " : "")
            << std::left << std::setw(7)
            << (G__AUTO==member_isstatic ? "auto " : "static ")
            << std::left << std::setw(24) << dname
            << " \"" << (char*)(comment ? comment : "(None)") << "\""
            << std::endl
            << std::setw(24) << std::left << "Cintex: declareField>"
            << "  Type:" 
            << std::left << std::setw(24) << ("[" + (dm.Properties().HasProperty("iotype") ? dm.Properties().PropertyAsString("iotype") : dmType.Name(SCOPED)) + "]")
            << " DeclBy:" << fClass.Name(SCOPED)
            << std::endl;
      }
      ::G__memvar_setup((void*)dm.Offset(),                         // p
                        member_type,                                // type
                        member_indir,                               // indirection
                        dmType.IsConst(),                        // const
                        member_tagnum,                              // tagnum
                        member_typnum,                              // typenum
                        member_isstatic,                            // statictype
                        member_access,                              // accessin
                        expr.c_str(),                               // expression
                        0,                                          // define macro
                        comment                                     // comment
                        ); 
   }

}}
