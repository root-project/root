// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingTypedefInfo
#define ROOT_TClingTypedefInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingTypedefInfo                                                    //
//                                                                      //
// Emulation of the CINT TypedefInfo class.                             //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// a typedef through the TypedefInfo class.  This class provides the    //
// same functionality, using an interface as close as possible to       //
// TypedefInfo but the typedef metadata comes from the Clang C++        //
// compiler, not CINT.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class tcling_TypedefInfo {
public:
   ~tcling_TypedefInfo();
   explicit tcling_TypedefInfo(cling::Interpreter*);
   explicit tcling_TypedefInfo(cling::Interpreter*, const char*);
   tcling_TypedefInfo(const tcling_TypedefInfo&);
   tcling_TypedefInfo& operator=(const tcling_TypedefInfo&);
   G__TypedefInfo* GetTypedefInfo() const;
   clang::Decl* GetDecl() const;
   void Init(const char* name);
   bool IsValidCint() const;
   bool IsValidClang() const;
   bool IsValid() const;
   int AdvanceToDecl(const clang::Decl*);
   int InternalNext();
   int Next();
   long Property() const;
   int Size() const;
   const char* TrueName() const;
   const char* Name() const;
   const char* Title() const;
private:
   //
   //  CINT info.
   //
   /// CINT typedef info for this class, we own.
   G__TypedefInfo* fTypedefInfo;
   //
   //  clang info.
   //
   /// Cling interpreter, we do *not* own.
   cling::Interpreter* fInterp;
   /// We need to skip the first increment to support the cint Next() semantics.
   bool fFirstTime;
   /// Flag for signaling the need to descend on this advancement.
   bool fDescend;
   /// Current decl in scope.
   clang::DeclContext::decl_iterator fIter;
   /// Current decl.
   clang::Decl* fDecl;
   /// Recursion stack for traversing nested scopes.
   std::vector<clang::DeclContext::decl_iterator> fIterStack;
};

#endif // ROOT_TClingTypedefInfo
