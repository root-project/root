// @(#)Root/meta:$Name:  $:$Id: TMethodCall.cxx,v 1.28 2006/11/24 14:24:54 rdm Exp $
// Author: Fons Rademakers   13/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMethodCall                                                          //
//                                                                      //
// Method or function calling interface. Objects of this class contain  //
// the (CINT) environment to call a global function or a method for an  //
// object of a specific class with the desired arguments. This class is //
// espicially useful when a method has to be called more times for      //
// different objects and/or with different arguments. If a function or  //
// method needs to be called only once one better uses                  //
// TInterpreter::Execute().                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TInterpreter.h"
#include "TMethodCall.h"
#include "TMethod.h"
#include "TClass.h"
#include "TROOT.h"
#include "Strlen.h"
#include "G__ci.h"
#include "Api.h"
#include "TVirtualMutex.h"
#include "TCint.h"

ClassImp(TMethodCall)

//______________________________________________________________________________
TMethodCall::TMethodCall()
{
   // Default TMethodCall ctor. Use Init() to initialize the method call
   // environment.

   fFunc     = 0;
   fOffset   = 0;
   fClass    = 0;
   fMetPtr   = 0;
   fMethod   = "";
   fParams   = "";
   fProto    = "";
   fDtorOnly = kFALSE;
   fRetType  = kNone;
}

//______________________________________________________________________________
TMethodCall::TMethodCall(TClass *cl, const char *method, const char *params)
{
   // Create a method invocation environment for a specific class, method and
   // parameters. The parameter string has the form: "\"aap\", 3, 4.35".
   // To execute the method call TMethodCall::Execute(object,...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   fFunc = 0;

   Init(cl, method, params);
}

//______________________________________________________________________________
TMethodCall::TMethodCall(const char *function, const char *params)
{
   // Create a global function invocation environment. The parameter
   // string has the form: "\"aap\", 3, 4,35". To execute the
   // function call TMethodCall::Execute(...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   fFunc = 0;

   Init(function, params);
}

//______________________________________________________________________________
TMethodCall::TMethodCall(const TMethodCall &orig) : TObject(orig)
{
   // Copy ctor.

   fFunc     = orig.fFunc ? new G__CallFunc(*orig.fFunc) : 0;
   fClass    = orig.fClass;
   fMethod   = orig.fMethod;
   fParams   = orig.fParams;
   fProto    = orig.fProto;
   fDtorOnly = orig.fDtorOnly;
   fRetType  = orig.fRetType;

   fMetPtr = 0;
}

//______________________________________________________________________________
TMethodCall &TMethodCall::operator=(const TMethodCall &rhs)
{
   // Assignement operator.

   if (this != &rhs) {
      delete fFunc;
      fFunc     = rhs.fFunc ? new G__CallFunc(*rhs.fFunc) : 0;
      fClass    = rhs.fClass;
      fMethod   = rhs.fMethod;
      fParams   = rhs.fParams;
      fProto    = rhs.fProto;
      fDtorOnly = rhs.fDtorOnly;
      fRetType  = rhs.fRetType;

      delete fMetPtr;
      fMetPtr = 0;
   }

   return *this;
}

//______________________________________________________________________________
TMethodCall::~TMethodCall()
{
   // TMethodCall dtor.

   delete fFunc;
   delete fMetPtr;
}

//______________________________________________________________________________
TObject *TMethodCall::Clone(const char *) const
{
   // Return an exact copy of this object.

   TObject *newobj = new TMethodCall(*this);
   return newobj;
}

//______________________________________________________________________________
static TClass *R__FindScope(const char *function, UInt_t &pos, G__ClassInfo &cinfo)
{
   // Helper function to find the scope associated with a qualified
   // function name

   if (function) {
      UInt_t nested = 0;
      for(int i=strlen(function); i>=0; --i) {
         switch(function[i]) {
            case '<': --nested; break;
            case '>': ++nested; break;
            case ':':
               if (nested==0) {
                  if (i>2 && function[i-1]==':') {
                     TString scope(function);
                     scope[i-1] = 0;
                     pos = i+1;
                     TClass *cl = TClass::GetClass(scope);
                     if (!cl) cinfo.Init(scope);
                     return cl;
                  }
               }
               break;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
void TMethodCall::Init(TClass *cl, const char *method, const char *params)
{
   // Initialize the method invocation environment. Necessary input
   // information: the class, method name and the parameter string
   // of the form "\"aap\", 3, 4.35".
   // To execute the method call TMethodCall::Execute(object,...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   G__ClassInfo cinfo;
   if (!cl) {
      UInt_t pos = 0;
      cl = R__FindScope(method,pos,cinfo);
      method = method+pos;
   }
   InitImplementation(method,params,0,cl,cinfo);
}

//______________________________________________________________________________
void TMethodCall::Init(const char *function, const char *params)
{
   // Initialize the function invocation environment. Necessary input
   // information: the function name and the parameter string of
   // the form "\"aap\", 3, 4.35".
   // To execute the method call TMethodCall::Execute(...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   UInt_t pos = 0;
   G__ClassInfo cinfo;
   TClass *cl = R__FindScope(function,pos,cinfo);
   InitImplementation(function+pos, params, 0, cl, cinfo);
}

//______________________________________________________________________________
void TMethodCall::InitImplementation(const char *methodname, const char *params,
                                     const char *proto, TClass *cl,
                                     const G__ClassInfo &cinfo) 
{
   // This function implements Init and InitWithPrototype.

   // 'methodname' should NOT have any scope information in it.  The scope
   // information should be passed via the TClass or G__ClassInfo.

   if (!fFunc)
      fFunc = new G__CallFunc;
   else
      fFunc->Init();

   fClass    = cl;
   fMetPtr   = 0;
   fMethod   = methodname;
   fParams   = params ? params : "";
   fProto    = proto ? proto : "";
   fDtorOnly = kFALSE;
   fRetType  = kNone;

   G__ClassInfo *scope = 0;
   G__ClassInfo global;
   if (cl) scope = cl->GetClassInfo();
   else scope = (G__ClassInfo*)&cinfo;
  
   if (!scope) return;

   R__LOCKGUARD2(gCINTMutex);
   if (params && params[0]) {
      fFunc->SetFunc(scope, (char *)methodname, (char *)params, &fOffset);
   } else if (proto && proto[0]) {
      fFunc->SetFuncProto(scope, (char *)methodname, (char *)proto, &fOffset);
   } else {
      // No parameters
      fFunc->SetFunc(scope, (char *)methodname, "", &fOffset);
   }

}

//______________________________________________________________________________
void TMethodCall::InitWithPrototype(TClass *cl, const char *method, const char *proto)
{
   // Initialize the method invocation environment. Necessary input
   // information: the class, method name and the prototype string of
   // the form: "char*,int,float".
   // To execute the method call TMethodCall::Execute(object,...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   G__ClassInfo cinfo;
   if (!cl) {
      UInt_t pos = 0;
      cl = R__FindScope(method,pos,cinfo);
      method = method+pos;
   }
   InitImplementation(method, 0, proto, cl, cinfo);
}

//______________________________________________________________________________
void TMethodCall::InitWithPrototype(const char *function, const char *proto)
{
   // Initialize the function invocation environment. Necessary input
   // information: the function name and the prototype string of
   // the form: "char*,int,float".
   // To execute the method call TMethodCall::Execute(...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   UInt_t pos = 0;
   G__ClassInfo cinfo;
   TClass *cl = R__FindScope(function,pos,cinfo);
   InitImplementation(function+pos, 0, proto, cl, cinfo);
}

//______________________________________________________________________________
Bool_t TMethodCall::IsValid() const
{
   // Return true if the method call has been properly initialized and is
   // usable.

   return fFunc ? fFunc->IsValid() : kFALSE;
}

//______________________________________________________________________________
TFunction *TMethodCall::GetMethod()
{
   // Returns the TMethod describing the method to be executed. This takes
   // all overriding and overloading into account (call TClass::GetMethod()).
   // Since finding the method is expensive the result is cached.

   // Since the object in the list of global function are often deleted
   // we need to copy them.

   if (!fMetPtr) {
      if (fClass) {
         if (fProto == "")
            fMetPtr = fClass->GetMethod(fMethod.Data(), fParams.Data());
         else
            fMetPtr = fClass->GetMethodWithPrototype(fMethod.Data(), fProto.Data());
         TMethod *met = dynamic_cast<TMethod*>(fMetPtr);
         if (met) fMetPtr = new TMethod(*met);
      } else {
         if (fProto == "")
            fMetPtr = gROOT->GetGlobalFunction(fMethod.Data(), fParams.Data(), kTRUE);
         else
            fMetPtr = gROOT->GetGlobalFunctionWithPrototype(fMethod.Data(), fProto.Data(), kTRUE);
         if (fMetPtr) fMetPtr = new TFunction(*fMetPtr);
      }
   }

   return fMetPtr;
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object)
{
   // Execute the method (with preset arguments) for the specified object.

   R__LOCKGUARD2(gCINTMutex);
   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   G__settemplevel(1);
   if (fDtorOnly) {
      Long_t saveglobalvar = G__getgvp();
      G__setgvp((Long_t)address);
      fFunc->Exec(address);
      G__setgvp(saveglobalvar);
   } else
      fFunc->Exec(address);
   G__settemplevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params)
{
   // Execute the method for the specified object and argument values.

   R__LOCKGUARD2(gCINTMutex);
   fFunc->SetArgs((char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   G__settemplevel(1);
   fFunc->Exec(address);
   G__settemplevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, Long_t &retLong)
{
   // Execute the method (with preset arguments) for the specified object.

   R__LOCKGUARD2(gCINTMutex);
   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   G__settemplevel(1);
   retLong = fFunc->ExecInt(address);
   G__settemplevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, Long_t &retLong)
{
   // Execute the method for the specified object and argument values.

   R__LOCKGUARD2(gCINTMutex);
   fFunc->SetArgs((char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   G__settemplevel(1);
   retLong = fFunc->ExecInt(address);
   G__settemplevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, Double_t &retDouble)
{
   // Execute the method (with preset arguments) for the specified object.

   R__LOCKGUARD2(gCINTMutex);
   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   G__settemplevel(1);
   retDouble = fFunc->ExecDouble(address);
   G__settemplevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, Double_t &retDouble)
{
   // Execute the method for the specified object and argument values.

   R__LOCKGUARD2(gCINTMutex);
   fFunc->SetArgs((char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   G__settemplevel(1);
   retDouble = fFunc->ExecDouble(address);
   G__settemplevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, char **retText)
{
   // Execute the method (with preset arguments) for the specified object.

   R__LOCKGUARD2(gCINTMutex);
   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   G__settemplevel(1);
   *retText =(char*) (fFunc->ExecInt(address));
   G__settemplevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, char **retText)
{
   // Execute the method for the specified object and argument values.

   R__LOCKGUARD2(gCINTMutex);
   fFunc->SetArgs((char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   G__settemplevel(1);
   *retText =(char*)(fFunc->ExecInt(address));
   G__settemplevel(-1);
}

//______________________________________________________________________________
TMethodCall::EReturnType TMethodCall::ReturnType()
{
   // Returns the return type of the method. Either (unsigned) long,
   // int, short and char, or float and double or anything else.
   // Since finding the return type is expensive the result is cached.

   if ( fRetType == kNone) {
      TFunction *func = GetMethod();
      if (func == 0) {
         fRetType = kOther;
         Error("ReturnType","Unknown method");
         return kOther;
      }

      // count the number of stars in the name.
      Int_t nstar = 0;
      const char* returntype = func->GetReturnTypeName();
      while (*returntype) {
         if (*returntype == '*') nstar++;
         returntype++;
      }

      G__TypedefInfo type(gInterpreter->TypeName(func->GetReturnTypeName()));
      const char *name = type.TrueName();

      Bool_t isEnum = kFALSE;
      if (!strcmp("(unknown)",name)) {
         G__TypeInfo type(func->GetReturnTypeName());
         name = type.TrueName();
         if (type.Property()&kIsEnum) {
            isEnum = kTRUE;
         }
      }

      if ((nstar==1) &&
          (!strcmp("unsigned char", name)        || !strcmp("char", name)         ||
           !strcmp("UChar_t", name)              || !strcmp("Char_t", name)       ||
           !strcmp("const unsigned char", name)  || !strcmp("const char", name)   ||
           !strcmp("const UChar_t", name)        || !strcmp("const Char_t", name) ||
           !strcmp("unsigned char*", name)       || !strcmp("char*", name)        ||
           !strcmp("UChar_t*", name)             || !strcmp("Char_t*", name)      ||
           !strcmp("const unsigned char*", name) || !strcmp("const char*", name)  ||
           !strcmp("const UChar_t*", name)       || !strcmp("const Char_t*", name)))
         fRetType = kString;
      else if (!strcmp("unsigned int", name)   || !strcmp("int", name)      ||
               !strcmp("unsigned long", name)  || !strcmp("long", name)     ||
               !strcmp("unsigned long long", name) || !strcmp("long long", name) ||
               !strcmp("unsigned short", name) || !strcmp("short", name)    ||
               !strcmp("unsigned char", name)  || !strcmp("char", name)     ||
               !strcmp("UInt_t", name)         || !strcmp("Int_t", name)    ||
               !strcmp("ULong_t", name)        || !strcmp("Long_t", name)   ||
               !strcmp("ULong64_t", name)      || !strcmp("Long_t64", name) ||
               !strcmp("UShort_t", name)       || !strcmp("Short_t", name)  ||
               !strcmp("UChar_t", name)        || !strcmp("Char_t", name)   ||
               !strcmp("Bool_t", name)         || !strcmp("bool", name)     ||
               strstr(name, "enum"))
         fRetType = kLong;
      else if (!strcmp("float", name)   || !strcmp("double", name)    ||
               !strcmp("Float_t", name) || !strcmp("Double_t", name))
         fRetType = kDouble;
      else if (isEnum)
         fRetType = kLong;
      else
         fRetType = kOther;
   }
   return fRetType;
}

//______________________________________________________________________________
void TMethodCall::SetParamPtrs(void *paramArr, Int_t nparam)
{
   // ParamArr is an array containing the function argument values.
   // If nparam = -1 then paramArr must contain values for all function
   // arguments, otherwise Nargs-NargsOpt <= nparam <= Nargs, where
   // Nargs is the number of all arguments and NargsOpt is the number
   // of default arguments.

   R__LOCKGUARD2(gCINTMutex);
   fFunc->SetArgArray((Long_t *)paramArr, nparam);
}

//______________________________________________________________________________
void TMethodCall::ResetParam()
{
   // Reset parameter list. To be used before the first call the SetParam().

   fFunc->ResetArg();
}

//______________________________________________________________________________
void TMethodCall::SetParam(Long_t l)
{
   // Set long method parameter.

   fFunc->SetArg(l);
}

//______________________________________________________________________________
void TMethodCall::SetParam(Double_t d)
{
   // Set double method parameter.

   fFunc->SetArg(d);
}

//______________________________________________________________________________
void TMethodCall::SetParam(Long64_t ll)
{
   // Set long long method parameter.

   fFunc->SetArg(ll);
}

//______________________________________________________________________________
void TMethodCall::SetParam(ULong64_t ull)
{
   // Set unsigned long long method parameter.

   fFunc->SetArg(ull);
}
