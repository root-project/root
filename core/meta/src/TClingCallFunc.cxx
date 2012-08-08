// @(#)root/core/meta:$Id$
// Author: Paul Russo   30/07/2012

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TClingCallFunc.h"

tcling_CallFunc::~tcling_CallFunc()
{
   delete fCallFunc;
   fCallFunc = 0;
   delete fMethod;
   fMethod = 0;
}

tcling_CallFunc::tcling_CallFunc(cling::Interpreter* interp)
   : fCallFunc(0), fInterp(interp), fMethod(0)
{
   fCallFunc = new G__CallFunc();
   fMethod = new tcling_MethodInfo(interp);
}

tcling_CallFunc::tcling_CallFunc(const tcling_CallFunc& rhs)
   : fCallFunc(0), fInterp(rhs.fInterp), fMethod(0)
{
   if (!rhs.IsValid()) {
      fCallFunc = new G__CallFunc();
      fMethod = new tcling_MethodInfo(rhs.fInterp);
      return;
   }
   fCallFunc = new G__CallFunc(*rhs.fCallFunc);
   fMethod = new tcling_MethodInfo(*rhs.fMethod);
}

tcling_CallFunc& tcling_CallFunc::operator=(const tcling_CallFunc& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   if (!rhs.IsValid()) {
      delete fCallFunc;
      fCallFunc = new G__CallFunc();
      fInterp = rhs.fInterp;
      delete fMethod;
      fMethod = new tcling_MethodInfo(rhs.fInterp);
   }
   else {
      delete fCallFunc;
      fCallFunc = new G__CallFunc(*rhs.fCallFunc);
      fInterp = rhs.fInterp;
      delete fMethod;
      fMethod = new tcling_MethodInfo(*rhs.fMethod);
   }
   return *this;
}

void tcling_CallFunc::Exec(void* address) const
{
   fCallFunc->Exec(address);
}

long tcling_CallFunc::ExecInt(void* address) const
{
   return fCallFunc->ExecInt(address);
}

long tcling_CallFunc::ExecInt64(void* address) const
{
   return fCallFunc->ExecInt64(address);
}

double tcling_CallFunc::ExecDouble(void* address) const
{
   return fCallFunc->ExecDouble(address);
}

void* tcling_CallFunc::FactoryMethod() const
{
   G__MethodInfo* info = new G__MethodInfo(fCallFunc->GetMethodInfo());
   tcling_MethodInfo* tcling_mi = new tcling_MethodInfo(fInterp, info);
   delete info;
   info = 0;
   return (void*) tcling_mi; // FIXME
}

void tcling_CallFunc::Init() const
{
   fCallFunc->Init();
}

G__InterfaceMethod tcling_CallFunc::InterfaceMethod() const
{
   return fCallFunc->InterfaceMethod();
}

bool tcling_CallFunc::IsValid() const
{
   return fCallFunc->IsValid();
}

void tcling_CallFunc::ResetArg() const
{
   fCallFunc->ResetArg();
}

void tcling_CallFunc::SetArg(long param) const
{
   fCallFunc->SetArg(param);
}

void tcling_CallFunc::SetArg(double param) const
{
   fCallFunc->SetArg(param);
}

void tcling_CallFunc::SetArg(long long param) const
{
   fCallFunc->SetArg(param);
}

void tcling_CallFunc::SetArg(unsigned long long param) const
{
   fCallFunc->SetArg(param);
}

void tcling_CallFunc::SetArgArray(long* paramArr, int nparam) const
{
   fCallFunc->SetArgArray(paramArr, nparam);
}

void tcling_CallFunc::SetArgs(const char* param) const
{
   fCallFunc->SetArgs(param);
}

void tcling_CallFunc::SetFunc(tcling_ClassInfo* info, const char* method, const char* params, long* offset) const
{
   fCallFunc->SetFunc(info->GetClassInfo(), method, params, offset);
}

void tcling_CallFunc::SetFunc(tcling_MethodInfo* info) const
{
   fCallFunc->SetFunc(*info->GetMethodInfo());
}

void tcling_CallFunc::SetFuncProto(tcling_ClassInfo* info, const char* method, const char* proto, long* offset) const
{
   fCallFunc->SetFuncProto(info->GetClassInfo(), method, proto, offset);
   //const clang::Decl* decl =
   //   fInterp->lookupFunctionProto(info->GetDecl(), method, proto);
}

