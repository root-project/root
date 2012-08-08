// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingBaseClassInfo
#define ROOT_TClingBaseClassInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingBaseClassInfo                                                  //
//                                                                      //
// Emulation of the CINT BaseClassInfo class.                           //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// the base classes of a class through the BaseClassInfo class.  This   //
// class provides the same functionality, using an interface as close   //
// as possible to BaseClassInfo but the base class metadata comes from  //
// the Clang C++ compiler, not CINT.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class tcling_BaseClassInfo {
public:
   ~tcling_BaseClassInfo();
   explicit tcling_BaseClassInfo(cling::Interpreter*); // NOT IMPLEMENTED.
   explicit tcling_BaseClassInfo(cling::Interpreter*, tcling_ClassInfo*);
   tcling_BaseClassInfo(const tcling_BaseClassInfo&);
   tcling_BaseClassInfo& operator=(const tcling_BaseClassInfo&);
   int InternalNext(int onlyDirect);
   bool IsValidCint() const;
   bool IsValidClang() const;
   bool IsValid() const;
   int Next();
   int Next(int onlyDirect);
   long Offset() const;
   long Property() const;
   long Tagnum() const;
   const char* FullName() const;
   const char* Name() const;
   const char* TmpltName() const;
private:
   //
   // CINT material.
   //
   /// CINT base class info, we own.
   G__BaseClassInfo* fBaseClassInfo;
   //
   // Cling material.
   //
   /// Cling interpreter, we do *not* own.
   cling::Interpreter* fInterp;
   /// Class we were intialized with, we own.
   tcling_ClassInfo* fClassInfo;
   /// Flag to provide Cint semantics for iterator advancement (not first time)
   bool fFirstTime;
   /// Flag for signaling the need to descend on this advancement.
   bool fDescend;
   /// Current class whose bases we are iterating through, we do *not* own.
   const clang::Decl* fDecl;
   /// Current iterator.
   clang::CXXRecordDecl::base_class_const_iterator fIter;
   /// Class info of base class our iterator is currently pointing at, we own.
   tcling_ClassInfo* fBaseInfo;
   /// Iterator stack.
   std::vector < std::pair < std::pair < const clang::Decl*,
       clang::CXXRecordDecl::base_class_const_iterator > , long > > fIterStack;
   /// Offset of the current base, fDecl, in the most-derived class.
   long fOffset;
};

#endif // ROOT_TClingBaseClassInfo
