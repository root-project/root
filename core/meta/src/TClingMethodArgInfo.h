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


namespace cling {
class Interpreter;
}

class TClingMethodInfo;
class TClingTypeInfo;

class TClingMethodArgInfo {

private:

   cling::Interpreter       *fInterp; // Cling interpreter, we do *not* own.
   const TClingMethodInfo   *fMethodInfo; // Function we return info about, we do *not* own.
   int                       fIdx; // Iterator, current parameter index.

public:

   explicit TClingMethodArgInfo(cling::Interpreter *interp) : fInterp(interp), fMethodInfo(0), fIdx(-1) {}
   TClingMethodArgInfo(cling::Interpreter *interp, const TClingMethodInfo *mi) : fInterp(interp), fMethodInfo(mi), fIdx(-1) {}

   bool                   IsValid() const;
   int                    Next();
   long                   Property() const;
   const char            *DefaultValue() const;
   const char            *Name() const;
   const TClingTypeInfo  *Type() const;
   const char            *TypeName() const;

};

#endif // ROOT_TClingMethodArgInfo
