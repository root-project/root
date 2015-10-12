// @(#)Root/meta:$Id$
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
#include "TVirtualMutex.h"
#include "TError.h"

ClassImp(TMethodCall)

//______________________________________________________________________________
TMethodCall::TMethodCall():
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   // Default TMethodCall ctor. Use Init() to initialize the method call
   // environment.
}

//______________________________________________________________________________
TMethodCall::TMethodCall(TClass *cl, CallFunc_t *callfunc, Long_t offset):
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   // Create a method invocation environment for a specific class, method
   // described by the callfunc.

   Init(cl, callfunc, offset);
}

//______________________________________________________________________________
TMethodCall::TMethodCall(TClass *cl, const char *method, const char *params):
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   // Create a method invocation environment for a specific class, method and
   // parameters. The parameter string has the form: "\"aap\", 3, 4.35".
   // To execute the method call TMethodCall::Execute(object,...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   Init(cl, method, params);
}

//______________________________________________________________________________
TMethodCall::TMethodCall(const char *function, const char *params):
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   // Create a global function invocation environment. The parameter
   // string has the form: "\"aap\", 3, 4,35". To execute the
   // function call TMethodCall::Execute(...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   Init(function, params);
}

//______________________________________________________________________________
TMethodCall::TMethodCall(const TFunction *func):
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   // Create a global function invocation environment base on a TFunction object.
   // To execute the function call TMethodCall::Execute(...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   Init(func);
}
//______________________________________________________________________________
TMethodCall::TMethodCall(const TMethodCall &orig) :
TObject(orig),
fFunc(orig.fFunc ? gCling->CallFunc_FactoryCopy(orig.fFunc) : 0),
fOffset(orig.fOffset), fClass(orig.fClass), fMetPtr(0 /*!*/),
fMethod(orig.fMethod), fParams(orig.fParams), fProto(orig.fProto),
fDtorOnly(orig.fDtorOnly), fRetType(orig.fRetType)
{
   // Copy ctor.
}

//______________________________________________________________________________
TMethodCall &TMethodCall::operator=(const TMethodCall &rhs)
{
   // Assignement operator.

   if (this != &rhs) {
      gCling->CallFunc_Delete(fFunc);
      fFunc     = rhs.fFunc ? gCling->CallFunc_FactoryCopy(rhs.fFunc) : 0;
      fOffset   = rhs.fOffset;
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

   gCling->CallFunc_Delete(fFunc);
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
// FIXME: We don't need to split that into lookup scope and lookup member.
// Consider merging the implementation with the new lookup functionality.
static TClass *R__FindScope(const char *function, UInt_t &pos, ClassInfo_t *cinfo)
{
   // Helper function to find the scope associated with a qualified
   // function name

   if (function) {
      UInt_t nested = 0;
      for(int i=strlen(function); i>=0; --i) {
         switch(function[i]) {
            case '<': ++nested; break;
            case '>': if (nested==0) { Error("TMethodCall R__FindScope","%s is not well formed function name",function); return 0; }
                      --nested; break;
            case ':':
               if (nested==0) {
                  if (i>1 && function[i-1]==':') {
                     TString scope(function);
                     scope[i-1] = 0;
                     pos = i+1;
                     TClass *cl = TClass::GetClass(scope);
                     if (!cl) gCling->ClassInfo_Init(cinfo, scope);
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
void TMethodCall::Init(TClass *cl, CallFunc_t *function, Long_t offset)
{
   // Initialize the method invocation environment based on
   // the CallFunc object and the TClass describing the function context.

   if (!function) {
      fOffset = 0;
      fDtorOnly = kFALSE;
      fRetType  = kNone;
      return;
   }

   MethodInfo_t* info = gCling->CallFunc_FactoryMethod(function);

   if (!fFunc)
      fFunc = gCling->CallFunc_Factory();
   else
      gCling->CallFunc_Init(fFunc);

   fClass = cl;
   if (fClass) {
      fMetPtr = new TMethod(info,cl);
   } else {
      fMetPtr = new TFunction(info);
   }
   fMethod = fMetPtr->GetName();
   fParams = "";
   fProto  = fMetPtr->GetSignature()+1; // skip leading )
   Ssiz_t s = fProto.Last(')');
   fProto.Remove(s); // still need to remove default values :(

   fOffset = offset;

   fDtorOnly = kFALSE;
   fRetType  = kNone;

   gCling->CallFunc_SetFunc(fFunc,info);

}

//______________________________________________________________________________
void TMethodCall::Init(const TFunction *function)
{
   // Initialize the method invocation environment based on
   // the TFunction object.

   if (!function) return;

   if (!fFunc)
      fFunc = gCling->CallFunc_Factory();
   else
      gCling->CallFunc_Init(fFunc);

   const TMethod *m = dynamic_cast<const TMethod*>(function);
   fClass = m ? m->GetClass() : 0;
   fMetPtr = (TFunction*)function->Clone();
   fMethod = fMetPtr->GetName();
   fParams = "";
   fProto  = fMetPtr->GetSignature()+1; // skip leading )
   Ssiz_t s = fProto.Last(')');
   fProto.Remove(s); // still need to remove default values :(

   fDtorOnly = kFALSE;
   fRetType  = kNone;

   gCling->CallFunc_SetFunc(fFunc,function->fInfo);

}

//______________________________________________________________________________
void TMethodCall::Init(TClass *cl, const char *method, const char *params, Bool_t objectIsConst /* = kFALSE */)
{
   // Initialize the method invocation environment. Necessary input
   // information: the class, method name and the parameter string
   // of the form "\"aap\", 3, 4.35".
   // To execute the method call TMethodCall::Execute(object,...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   ClassInfo_t *cinfo = gCling->ClassInfo_Factory();
   if (!cl) {
      UInt_t pos = 0;
      cl = R__FindScope(method,pos,cinfo);
      method = method+pos;
   }
   InitImplementation(method,params,0,objectIsConst,cl,cinfo);
   gCling->ClassInfo_Delete(cinfo);
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
   ClassInfo_t *cinfo = gCling->ClassInfo_Factory();
   TClass *cl = R__FindScope(function,pos,cinfo);
   InitImplementation(function+pos, params, 0, false, cl, cinfo);
   gCling->ClassInfo_Delete(cinfo);
}

//______________________________________________________________________________
void TMethodCall::InitImplementation(const char *methodname, const char *params,
                                     const char *proto,
                                     Bool_t objectIsConst, TClass *cl,
                                     const ClassInfo_t *cinfo,
                                     ROOT::EFunctionMatchMode mode /* = ROOT::kConversionMatch */)
{
   // This function implements Init and InitWithPrototype.

   // 'methodname' should NOT have any scope information in it.  The scope
   // information should be passed via the TClass or CINT ClassInfo.

   if (!fFunc) {
      R__LOCKGUARD2(gInterpreterMutex);
      fFunc = gCling->CallFunc_Factory();
   } else
      gCling->CallFunc_Init(fFunc);

   fClass    = cl;
   fMetPtr   = 0;
   fMethod   = methodname;
   fParams   = params ? params : "";
   fProto    = proto ? proto : "";
   fDtorOnly = kFALSE;
   fRetType  = kNone;

   ClassInfo_t *scope = 0;
   if (cl) scope = (ClassInfo_t*)cl->GetClassInfo();
   else    scope = (ClassInfo_t*)cinfo;

   if (!scope) return;

   R__LOCKGUARD2(gInterpreterMutex);
   if (params && params[0]) {
      gCling->CallFunc_SetFunc(fFunc, scope, (char *)methodname, (char *)params, objectIsConst, &fOffset);
   } else if (proto && proto[0]) {
      gCling->CallFunc_SetFuncProto(fFunc, scope, (char *)methodname, (char *)proto, objectIsConst, &fOffset, mode);
   } else {
      // No parameters
      gCling->CallFunc_SetFuncProto(fFunc, scope, (char *)methodname, "", objectIsConst, &fOffset, mode);
   }
}

//______________________________________________________________________________
void TMethodCall::InitWithPrototype(TClass *cl, const char *method, const char *proto, Bool_t objectIsConst /* = kFALSE */, ROOT::EFunctionMatchMode mode /* = ROOT::kConversionMatch */)
{
   // Initialize the method invocation environment. Necessary input
   // information: the class, method name and the prototype string of
   // the form: "char*,int,float".
   // To execute the method call TMethodCall::Execute(object,...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   ClassInfo_t *cinfo = gCling->ClassInfo_Factory();
   if (!cl) {
      UInt_t pos = 0;
      cl = R__FindScope(method,pos,cinfo);
      method = method+pos;
   }
   InitImplementation(method, 0, proto, objectIsConst, cl, cinfo, mode);
   gCling->ClassInfo_Delete(cinfo);
}

//______________________________________________________________________________
void TMethodCall::InitWithPrototype(const char *function, const char *proto, ROOT::EFunctionMatchMode mode /* = ROOT::kConversionMatch */)
{
   // Initialize the function invocation environment. Necessary input
   // information: the function name and the prototype string of
   // the form: "char*,int,float".
   // To execute the method call TMethodCall::Execute(...).
   // This two step method is much more efficient than calling for
   // every invocation TInterpreter::Execute(...).

   UInt_t pos = 0;
   ClassInfo_t *cinfo = gCling->ClassInfo_Factory();
   TClass *cl = R__FindScope(function,pos,cinfo);
   InitImplementation(function+pos, 0, proto, false, cl, cinfo, mode);
   gCling->ClassInfo_Delete(cinfo);
}

//______________________________________________________________________________
Bool_t TMethodCall::IsValid() const
{
   // Return true if the method call has been properly initialized and is
   // usable.

   return fFunc ? gCling->CallFunc_IsValid(fFunc) : kFALSE;
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
      if (fFunc && gCling->CallFunc_IsValid(fFunc)) {
         if (fClass) {
            fMetPtr = new TMethod( gCling->CallFunc_FactoryMethod(fFunc), fClass );
         } else {
            fMetPtr = new TFunction( gCling->CallFunc_FactoryMethod(fFunc) );
         }
      } else if (fClass) {
         if (fProto == "") {
            fMetPtr = fClass->GetMethod(fMethod.Data(), fParams.Data());
         } else {
            fMetPtr = fClass->GetMethodWithPrototype(fMethod.Data(), fProto.Data());
         }
         TMethod *met = dynamic_cast<TMethod*>(fMetPtr);
         if (met) fMetPtr = new TMethod(*met);
      } else {
         if (fProto == "")
            fMetPtr = gROOT->GetGlobalFunction(fMethod.Data(), fParams.Data(), kFALSE);
         else
            fMetPtr = gROOT->GetGlobalFunctionWithPrototype(fMethod.Data(), fProto.Data(), kFALSE);
         if (fMetPtr) fMetPtr = new TFunction(*fMetPtr);
      }
   }

   return fMetPtr;
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object)
{
   // Execute the method (with preset arguments) for the specified object.

   if (!fFunc) return;

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   if (!fDtorOnly && fMethod[0]=='~') {
      Error("Execute","TMethodCall can no longer be use to call the operator delete and the destructor at the same time");
   }
   gCling->CallFunc_Exec(fFunc,address);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params)
{
   // Execute the method for the specified object and argument values.

   if (!fFunc) return;

   // SetArgs contains the necessary lock.
   gCling->CallFunc_SetArgs(fFunc, (char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   gCling->SetTempLevel(1);
   gCling->CallFunc_Exec(fFunc,address);
   gCling->SetTempLevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, Long_t &retLong)
{
   // Execute the method (with preset arguments) for the specified object.

   if (!fFunc) return;

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   gCling->SetTempLevel(1);
   retLong = gCling->CallFunc_ExecInt(fFunc,address);
   gCling->SetTempLevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, Long_t &retLong)
{
   // Execute the method for the specified object and argument values.

   if (!fFunc) return;

   // SetArgs contains the necessary lock.
   gCling->CallFunc_SetArgs(fFunc, (char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   gCling->SetTempLevel(1);
   retLong = gCling->CallFunc_ExecInt(fFunc,address);
   gCling->SetTempLevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, Double_t &retDouble)
{
   // Execute the method (with preset arguments) for the specified object.

   if (!fFunc) return;

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   gCling->SetTempLevel(1);
   retDouble = gCling->CallFunc_ExecDouble(fFunc,address);
   gCling->SetTempLevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, Double_t &retDouble)
{
   // Execute the method for the specified object and argument values.

   if (!fFunc) return;

   gCling->CallFunc_SetArgs(fFunc, (char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   gCling->SetTempLevel(1);
   retDouble = gCling->CallFunc_ExecDouble(fFunc,address);
   gCling->SetTempLevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, char **retText)
{
   // Execute the method (with preset arguments) for the specified object.

   if (!fFunc) return;

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   gCling->SetTempLevel(1);
   *retText =(char*) (gCling->CallFunc_ExecInt(fFunc,address));
   gCling->SetTempLevel(-1);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, char **retText)
{
   // Execute the method for the specified object and argument values.

   if (!fFunc) return;

   // SetArgs contains the necessary lock.
   gCling->CallFunc_SetArgs(fFunc, (char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   gCling->SetTempLevel(1);
   *retText =(char*)(gCling->CallFunc_ExecInt(fFunc,address));
   gCling->SetTempLevel(-1);
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

      fRetType = gCling->MethodCallReturnType(func);
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

   if (!fFunc) return;
   gCling->CallFunc_SetArgArray(fFunc,(Long_t *)paramArr, nparam);
}

//______________________________________________________________________________
void TMethodCall::ResetParam()
{
   // Reset parameter list. To be used before the first call the SetParam().

   if (!fFunc) return;
   gCling->CallFunc_ResetArg(fFunc);
}

//______________________________________________________________________________
void TMethodCall::SetParam(Long_t l)
{
   // Add a long method parameter.

   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,l);
}

//______________________________________________________________________________
void TMethodCall::SetParam(Float_t f)
{
   // Add a double method parameter.

   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,f);
}

//______________________________________________________________________________
void TMethodCall::SetParam(Double_t d)
{
   // Add a double method parameter.

   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,d);
}

//______________________________________________________________________________
void TMethodCall::SetParam(Long64_t ll)
{
   // Add a long long method parameter.

   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,ll);
}

//______________________________________________________________________________
void TMethodCall::SetParam(ULong64_t ull)
{
   // Add a unsigned long long method parameter.

   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,ull);
}
