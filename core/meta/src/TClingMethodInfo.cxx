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

#include "TClingMethodInfo.h"

tcling_MethodInfo::~tcling_MethodInfo()
{
   delete fMethodInfo;
   fMethodInfo = 0;
   delete fInitialClassInfo;
   fInitialClassInfo = 0;
   fDecl = 0;
   fFunction = 0;
}

tcling_MethodInfo::tcling_MethodInfo(cling::Interpreter* interp)
   : fMethodInfo(0)
   , fInterp(interp)
   , fInitialClassInfo(0)
   , fDecl(0)
   , fFunction(0)
{
   fMethodInfo = new G__MethodInfo();
   fInitialClassInfo = new tcling_ClassInfo(fInterp);
}

tcling_MethodInfo::tcling_MethodInfo(cling::Interpreter* interp,
                                     G__MethodInfo* info)
   : fMethodInfo(0)
   , fInterp(interp)
   , fInitialClassInfo(0)
   , fDecl(0)
   , fFunction(0)
{
   fMethodInfo = new G__MethodInfo(*info);
}

tcling_MethodInfo::tcling_MethodInfo(cling::Interpreter* interp,
                                     tcling_ClassInfo* tcling_class_info)
   : fMethodInfo(0)
   , fInterp(interp)
   , fInitialClassInfo(0)
   , fDecl(0)
   , fFunction(0)
{
   if (!tcling_class_info || !tcling_class_info->IsValid()) {
      fMethodInfo = new G__MethodInfo();
      fInitialClassInfo = new tcling_ClassInfo(fInterp);
      return;
   }
   fMethodInfo = new G__MethodInfo();
   fMethodInfo->Init(*tcling_class_info->GetClassInfo());
   fInitialClassInfo = new tcling_ClassInfo(*tcling_class_info);
}

tcling_MethodInfo::tcling_MethodInfo(const tcling_MethodInfo& rhs)
   : fMethodInfo(0)
   , fInterp(rhs.fInterp)
   , fInitialClassInfo(0)
   , fDecl(0)
   , fFunction(0)
{
   if (!rhs.IsValid()) {
      fMethodInfo = new G__MethodInfo();
      fInitialClassInfo = new tcling_ClassInfo(fInterp);
      return;
   }
   fMethodInfo = new G__MethodInfo(*rhs.fMethodInfo);
   fInitialClassInfo = new tcling_ClassInfo(*rhs.fInitialClassInfo);
   fDecl = rhs.fDecl;
   fIter = rhs.fIter;
   fFunction = rhs.fFunction;
}

tcling_MethodInfo& tcling_MethodInfo::operator=(const tcling_MethodInfo& rhs)
{
   if (this == &rhs) {
      return *this;
   }
   if (!rhs.IsValid()) {
      delete fMethodInfo;
      fMethodInfo = new G__MethodInfo();
      fInterp = rhs.fInterp;
      delete fInitialClassInfo;
      fInitialClassInfo = new tcling_ClassInfo(fInterp);
      fDecl = 0;
      fFunction = 0;
   }
   else {
      delete fMethodInfo;
      fMethodInfo = new G__MethodInfo(*rhs.fMethodInfo);
      fInterp = rhs.fInterp;
      delete fInitialClassInfo;
      fInitialClassInfo = new tcling_ClassInfo(*rhs.fInitialClassInfo);
      fDecl = rhs.fDecl;
      fIter = rhs.fIter;
      fFunction = rhs.fFunction;
   }
   return *this;
}

G__MethodInfo* tcling_MethodInfo::GetMethodInfo() const
{
   return fMethodInfo;
}

void tcling_MethodInfo::CreateSignature(TString& signature) const
{
   G__MethodArgInfo arg(*fMethodInfo);
   int ifirst = 0;
   signature = "(";
   while (arg.Next()) {
      if (ifirst) {
         signature += ", ";
      }
      if (arg.Type() == 0) {
         break;
      }
      signature += arg.Type()->Name();
      if (arg.Name() && strlen(arg.Name())) {
         signature += " ";
         signature += arg.Name();
      }
      if (arg.DefaultValue()) {
         signature += " = ";
         signature += arg.DefaultValue();
      }
      ifirst++;
   }
   signature += ")";
}

void tcling_MethodInfo::Init(clang::Decl* decl)
{
   clang::DeclContext* DC = decl->getDeclContext()->getPrimaryContext();
   fDecl = llvm::cast<clang::Decl>(DC);
   fInitialClassInfo = new tcling_ClassInfo(fInterp, fDecl);
   fIter = clang::DeclContext::specific_decl_iterator<clang::FunctionDecl>(
      clang::DeclContext::decl_iterator(decl));
   fFunction = decl;
}

G__InterfaceMethod tcling_MethodInfo::InterfaceMethod() const
{
   G__InterfaceMethod p = fMethodInfo->InterfaceMethod();
   if (!p) {
      struct G__bytecodefunc* bytecode = fMethodInfo->GetBytecode();
      if (bytecode) {
         p = (G__InterfaceMethod) G__exec_bytecode;
      }
   }
   return p;
}

bool tcling_MethodInfo::IsValid() const
{
   return fMethodInfo->IsValid();
}

int tcling_MethodInfo::NArg() const
{
   return fMethodInfo->NArg();
}

int tcling_MethodInfo::NDefaultArg() const
{
   return fMethodInfo->NDefaultArg();
}

int tcling_MethodInfo::Next() const
{
   return fMethodInfo->Next();
}

long tcling_MethodInfo::Property() const
{
   return fMethodInfo->Property();
}

void* tcling_MethodInfo::Type() const
{
   return fMethodInfo->Type();
}

const char* tcling_MethodInfo::GetMangledName() const
{
   return fMethodInfo->GetMangledName();
}

const char* tcling_MethodInfo::GetPrototype() const
{
   return fMethodInfo->GetPrototype();
}

const char* tcling_MethodInfo::Name() const
{
   return fMethodInfo->Name();
}

const char* tcling_MethodInfo::TypeName() const
{
   return fMethodInfo->Type()->Name();
}

const char* tcling_MethodInfo::Title() const
{
   return fMethodInfo->Title();
}

