// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CallFunc
#define ROOT_CallFunc

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClingCallFunc                                                       //
//                                                                      //
// Emulation of the CINT CallFunc class.                                //
//                                                                      //
// The CINT C++ interpreter provides an interface for calling           //
// functions through the generated wrappers in dictionaries with        //
// the CallFunc class. This class provides the same functionality,      //
// using an interface as close as possible to CallFunc but the          //
// function metadata and calling service comes from the Cling           //
// C++ interpreter and the Clang C++ compiler, not CINT.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class tcling_CallFunc {
public:
   ~tcling_CallFunc();
   explicit tcling_CallFunc(cling::Interpreter*);
   tcling_CallFunc(const tcling_CallFunc&);
   tcling_CallFunc& operator=(const tcling_CallFunc&);
   void Exec(void* address) const;
   long ExecInt(void* address) const;
   long ExecInt64(void* address) const;
   double ExecDouble(void* address) const;
   void* FactoryMethod() const;
   void Init() const;
   G__InterfaceMethod InterfaceMethod() const;
   bool IsValid() const;
   void ResetArg() const;
   void SetArg(long param) const;
   void SetArg(double param) const;
   void SetArg(long long param) const;
   void SetArg(unsigned long long param) const;
   void SetArgArray(long* paramArr, int nparam) const;
   void SetArgs(const char* param) const;
   void SetFunc(tcling_ClassInfo* info, const char* method, const char* params, long* offset) const;
   void SetFunc(tcling_MethodInfo* info) const;
   void SetFuncProto(tcling_ClassInfo* info, const char* method, const char* proto, long* offset) const;
private:
   //
   // CINT material.
   //
   /// cint method iterator, we own.
   G__CallFunc* fCallFunc;
   //
   // Cling material.
   //
   /// Cling interpreter, we do *not* own.
   cling::Interpreter* fInterp;
   /// Current method.
   tcling_MethodInfo* fMethod;
};

#endif // ROOT_CallFunc
