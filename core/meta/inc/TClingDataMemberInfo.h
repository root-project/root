// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingDataMemberInfo
#define ROOT_TClingDataMemberInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingDataMemberInfo                                                 //
//                                                                      //
// Emulation of the CINT DataMemberInfo class.                          //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// the data members of a class through the DataMemberInfo class.  This  //
// class provides the same functionality, using an interface as close   //
// as possible to DataMemberInfo but the data member metadata comes     //
// from the Clang C++ compiler, not CINT.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TClingDataMemberInfo {
public:
   ~TClingDataMemberInfo();
   explicit TClingDataMemberInfo(cling::Interpreter*);
   TClingDataMemberInfo(cling::Interpreter*, tcling_ClassInfo*);
   TClingDataMemberInfo(const TClingDataMemberInfo&);
   TClingDataMemberInfo& operator=(const TClingDataMemberInfo&);
   G__DataMemberInfo* GetDataMemberInfo() const;
   G__ClassInfo* GetClassInfo() const;
   tcling_ClassInfo* GetTClingClassInfo() const;
   clang::Decl* GetDecl() const;
   int ArrayDim() const;
   bool IsValidCint() const;
   bool IsValidClang() const;
   bool IsValid() const;
   int MaxIndex(int dim) const;
   int InternalNext();
   bool Next();
   long Offset() const;
   long Property() const;
   long TypeProperty() const;
   int TypeSize() const;
   const char* TypeName() const;
   const char* TypeTrueName() const;
   const char* Name() const;
   const char* Title() const;
   const char* ValidArrayIndex() const;
private:
   //
   // CINT material.
   //
   /// CINT data member info, we own.
   G__DataMemberInfo* fDataMemberInfo;
   /// CINT class info, we own.
   G__ClassInfo* fClassInfo;
   //
   // Clang material.
   //
   /// Cling interpreter, we do *not* own.
   cling::Interpreter* fInterp;
   /// Class we are iterating over, we own.
   tcling_ClassInfo* fTClingClassInfo;
   /// We need to skip the first increment to support the cint Next() semantics.
   bool fFirstTime;
   /// Current decl.
   clang::DeclContext::decl_iterator fIter;
   /// Recursion stack for traversing nested transparent scopes.
   std::vector<clang::DeclContext::decl_iterator> fIterStack;
};

#endif // ROOT_TClingDataMemberInfo
