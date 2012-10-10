// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingClassInfo
#define ROOT_TClingClassInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingClassInfo                                                      //
//                                                                      //
// Emulation of the CINT ClassInfo class.                               //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a class through the ClassInfo class.  This class provides the same   //
// functionality, using an interface as close as possible to ClassInfo  //
// but the class metadata comes from the Clang C++ compiler, not CINT.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClingMethodInfo.h"

#include "clang/AST/DeclBase.h"

#include <vector>
#include <string>

namespace cling {
class Interpreter;
}

namespace clang {
class Decl;
}

class TClingClassInfo {

private:

   cling::Interpreter   *fInterp; // Cling interpreter, we do *not* own.
   bool                  fFirstTime; // We need to skip the first increment to support the cint Next() semantics.
   bool                  fDescend; // Flag for signaling the need to descend on this advancement.
   clang::DeclContext::decl_iterator fIter; // Current decl in scope.
   const clang::Decl    *fDecl; // Current decl, we do *not* own.
   std::vector<clang::DeclContext::decl_iterator> fIterStack; // Recursion stack for traversing nested scopes.
   std::string           fTitle; // The meta info for the class.
public: // Types

   enum MatchMode {
      ExactMatch = 0,
      ConversionMatch = 1,
      ConversionMatchBytecode = 2
   };
   enum InheritanceMode {
      InThisScope = 0,
      WithInheritance = 1
   };

public:

   explicit TClingClassInfo(); // NOT IMPLEMENTED
   explicit TClingClassInfo(cling::Interpreter *);
   explicit TClingClassInfo(cling::Interpreter *, const char *);
   explicit TClingClassInfo(cling::Interpreter *, const clang::Decl *);

   const clang::Decl   *GetDecl() const { return fDecl; }
   long                 ClassProperty() const;
   void                 Delete(void *arena) const;
   void                 DeleteArray(void *arena, bool dtorOnly) const;
   void                 Destruct(void *arena) const;
   TClingMethodInfo     GetMethod(const char *fname, const char *proto,
                                  long *poffset, MatchMode mode = ConversionMatch,
                                  InheritanceMode imode = WithInheritance) const;
   int                  GetMethodNArg(const char *method, const char *proto) const;
   bool                 HasDefaultConstructor() const;
   bool                 HasMethod(const char *name) const;
   void                 Init(const char *name);
   void                 Init(int tagnum);
   bool                 IsBase(const char *name) const;
   static bool          IsEnum(cling::Interpreter *interp, const char *name);
   bool                 IsLoaded() const;
   bool                 IsValid() const;
   bool                 IsValidMethod(const char *method, const char *proto, long *offset) const;
   int                  AdvanceToDecl(const clang::Decl *);
   int                  InternalNext();
   int                  Next();
   void                *New() const;
   void                *New(int n) const;
   void                *New(int n, void *arena) const;
   void                *New(void *arena) const;
   long                 Property() const;
   int                  RootFlag() const;
   int                  Size() const;
   long                 Tagnum() const;
   const char          *FileName() const;
   const char          *FullName() const;
   const char          *Name() const;
   const char          *Title();
   const char          *TmpltName() const;

};

#endif // ROOT_TClingClassInfo
