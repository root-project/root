// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClingMethodArgInfo
#define ROOT_TClingMethodArgInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingMethodArgInfo                                                  //
//                                                                      //
// Emulation of the CINT MethodInfo class.                              //
//                                                                      //
// The CINT C++ interpreter provides an interface to metadata about     //
// the arguments to a function through the MethodArgInfo class.  This   //
// class provides the same functionality, using an interface as close   //
// as possible to MethodArgInfo but the typedef metadata comes from     //
// the Clang C++ compiler, not CINT.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class tcling_MethodArgInfo {
public:
   ~tcling_MethodArgInfo();
   explicit tcling_MethodArgInfo(cling::Interpreter*);
   explicit tcling_MethodArgInfo(cling::Interpreter*, const tcling_MethodInfo*);
   tcling_MethodArgInfo(const tcling_MethodArgInfo&);
   tcling_MethodArgInfo& operator=(const tcling_MethodArgInfo&);
   G__MethodInfo* GetMethodArgInfo() const;
   bool IsValidClang() const;
   bool IsValidCint() const;
   bool IsValid() const;
   int Next();
   long Property() const;
   const char* DefaultValue() const;
   const char* Name() const;
   const tcling_TypeInfo* Type() const;
   const char* TypeName() const;
private:
   //
   // CINT material.
   //
   /// cint method argument iterator, we own.
   G__MethodArgInfo* fMethodArgInfo;
   //
   // Cling material.
   //
   /// Cling interpreter, we do *not* own.
   cling::Interpreter* fInterp;
   /// Function we return info about, we do *not* own.
   const tcling_MethodInfo* fMethodInfo;
   /// Iterator, current parameter index.
   int fIdx;
};

#endif // ROOT_TClingMethodArgInfo
