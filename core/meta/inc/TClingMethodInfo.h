// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingMethodInfo
#define ROOT_TClingMethodInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingMethodInfo                                                     //
//                                                                      //
// Emulation of the CINT MethodInfo class.                              //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a function through the MethodInfo class.  This class provides the    //
// same functionality, using an interface as close as possible to       //
// MethodInfo but the typedef metadata comes from the Clang C++         //
// compiler, not CINT.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class tcling_MethodInfo {
public:
   ~tcling_MethodInfo();
   explicit tcling_MethodInfo(cling::Interpreter*);
   explicit tcling_MethodInfo(cling::Interpreter*, G__MethodInfo* info); // FIXME
   explicit tcling_MethodInfo(cling::Interpreter*, tcling_ClassInfo*);
   tcling_MethodInfo(const tcling_MethodInfo&);
   tcling_MethodInfo& operator=(const tcling_MethodInfo&);
   G__MethodInfo* GetMethodInfo() const;
   void CreateSignature(TString& signature) const;
   void Init(clang::Decl*);
   G__InterfaceMethod InterfaceMethod() const;
   bool IsValid() const;
   int NArg() const;
   int NDefaultArg() const;
   int Next() const;
   long Property() const;
   void* Type() const;
   const char* GetMangledName() const;
   const char* GetPrototype() const;
   const char* Name() const;
   const char* TypeName() const;
   const char* Title() const;
private:
   //
   // CINT material.
   //
   /// cint method iterator, we own.
   G__MethodInfo* fMethodInfo;
   //
   // Cling material.
   //
   /// Cling interpreter, we do *not* own.
   cling::Interpreter* fInterp;
   /// Class, namespace, or translation unit we were initialized with.
   tcling_ClassInfo* fInitialClassInfo;
   /// Class, namespace, or translation unit we are iterating over now.
   clang::Decl* fDecl;
   /// Our iterator.
   clang::DeclContext::specific_decl_iterator<clang::FunctionDecl> fIter;
   /// Our iterator's current function.
   clang::Decl* fFunction;
};

#endif // ROOT_TClingMethodInfo
