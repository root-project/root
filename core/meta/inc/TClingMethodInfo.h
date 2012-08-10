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

class TClingMethodInfo {
public:
   ~TClingMethodInfo();
   explicit TClingMethodInfo(cling::Interpreter*);
   // FIXME: We need this only for cint support, remove when cint is gone.
   explicit TClingMethodInfo(cling::Interpreter*, G__MethodInfo* info);
   explicit TClingMethodInfo(cling::Interpreter*, tcling_ClassInfo*);
   TClingMethodInfo(const TClingMethodInfo&);
   TClingMethodInfo& operator=(const TClingMethodInfo&);
   G__MethodInfo* GetMethodInfo() const;
   const clang::FunctionDecl* GetMethodDecl() const;
   void CreateSignature(TString& signature) const;
   void Init(const clang::FunctionDecl*);
   void* InterfaceMethod() const;
   bool IsValidCint() const;
   bool IsValidClang() const;
   bool IsValid() const;
   int NArg() const;
   int NDefaultArg() const;
   int InternalNext();
   int Next();
   long Property() const;
   tcling_TypeInfo* Type() const;
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
   /// Set of DeclContext that we will iterate over.
   llvm::SmallVector<clang::DeclContext*, 2> fContexts;
   /// Flag for first time incrementing iterator, cint semantics are weird.
   bool fFirstTime;
   /// Index in fContexts of DeclContext we are iterating over.
   unsigned int fContextIdx;
   /// Our iterator.
   clang::DeclContext::decl_iterator fIter;
};

#endif // ROOT_TClingMethodInfo
