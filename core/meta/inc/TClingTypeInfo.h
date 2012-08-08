// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingTypeInfo
#define ROOT_TClingTypeInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingTypeInfo                                                       //
//                                                                      //
// Emulation of the CINT TypeInfo class.                                //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a type through the TypeInfo class.  This class provides the same     //
// functionality, using an interface as close as possible to TypeInfo   //
// but the type metadata comes from the Clang C++ compiler, not CINT.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TClingTypeInfo {
public:
   ~TClingTypeInfo();
   explicit TClingTypeInfo(cling::Interpreter*);
   explicit TClingTypeInfo(cling::Interpreter*, const char* name);
   TClingTypeInfo(const TClingTypeInfo&);
   TClingTypeInfo& operator=(const TClingTypeInfo&);
   G__TypeInfo* GetTypeInfo() const;
   G__ClassInfo* GetClassInfo() const;
   clang::Decl* GetDecl() const;
   void Init(const char* name);
   bool IsValid() const;
   const char* Name() const;
   long Property() const;
   int RefType() const;
   int Size() const;
   const char* TrueName() const;
private:
   //
   //  CINT part
   //
   /// CINT type info, we own.
   G__TypeInfo* fTypeInfo;
   /// CINT class info, we own.
   G__ClassInfo* fClassInfo;
   //
   //  Cling part
   //
   /// Cling interpreter, we do *not* own.
   cling::Interpreter* fInterp;
   /// Clang AST Node for the type, we do *not* own.
   clang::Decl* fDecl;
};

#endif // ROOT_TClingTypeInfo
