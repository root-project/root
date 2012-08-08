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
   long long ExecInt64(void* address) const;
   double ExecDouble(void* address) const;
   tcling_MethodInfo* FactoryMethod() const;
   void Init();
   void* InterfaceMethod() const;
   bool IsValidCint() const;
   bool IsValidClang() const;
   bool IsValid() const;
   void ResetArg();
   void SetArg(long param);
   void SetArg(double param);
   void SetArg(long long param);
   void SetArg(unsigned long long param);
   void SetArgArray(long* paramArr, int nparam);
   void SetArgs(const char* params);
   void SetFunc(const tcling_ClassInfo* info, const char* method, const char* params, long* offset);
   void SetFunc(const tcling_MethodInfo* info);
   void SetFuncProto(const tcling_ClassInfo* info, const char* method, const char* proto, long* offset);
   void Init(const clang::FunctionDecl*);
   llvm::GenericValue Invoke(const std::vector<llvm::GenericValue>& ArgValues) const;
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
   /// Current method, we own.
   tcling_MethodInfo* fMethod;
   /// Execution Engine function for current method, we do *not* own.
   llvm::Function* fEEFunc;
   /// Pointer to actual compiled code, we do *not* own.
   void* fEEAddr;
   /// Arguments to pass to function.
   std::vector<llvm::GenericValue> fArgs;
};

#endif // ROOT_CallFunc
