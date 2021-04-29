// @(#)Root/meta:$Id$
// Author: Fons Rademakers   13/06/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TMethodCall
Method or function calling interface. Objects of this class contain
the (CINT) environment to call a global function or a method for an
object of a specific class with the desired arguments. This class is
especially useful when a method has to be called more times for
different objects and/or with different arguments. If a function or
method needs to be called only once one better uses
TInterpreter::Execute().
*/

#include "TInterpreter.h"
#include "TMethodCall.h"
#include "TMethod.h"
#include "TClass.h"
#include "TROOT.h"
#include "Strlen.h"
#include "TVirtualMutex.h"
#include "TError.h"

ClassImp(TMethodCall);

////////////////////////////////////////////////////////////////////////////////
/// Default TMethodCall ctor. Use Init() to initialize the method call
/// environment.

TMethodCall::TMethodCall():
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create a method invocation environment for a specific class, method
/// described by the callfunc.

TMethodCall::TMethodCall(TClass *cl, CallFunc_t *callfunc, Longptr_t offset):
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   Init(cl, callfunc, offset);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a method invocation environment for a specific class, method and
/// parameters. The parameter string has the form: "\"aap\", 3, 4.35".
/// To execute the method call TMethodCall::Execute(object,...).
/// This two step method is much more efficient than calling for
/// every invocation TInterpreter::Execute(...).

TMethodCall::TMethodCall(TClass *cl, const char *method, const char *params):
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   Init(cl, method, params);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a global function invocation environment. The parameter
/// string has the form: "\"aap\", 3, 4,35". To execute the
/// function call TMethodCall::Execute(...).
/// This two step method is much more efficient than calling for
/// every invocation TInterpreter::Execute(...).

TMethodCall::TMethodCall(const char *function, const char *params):
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   Init(function, params);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a global function invocation environment base on a TFunction object.
/// To execute the function call TMethodCall::Execute(...).
/// This two step method is much more efficient than calling for
/// every invocation TInterpreter::Execute(...).

TMethodCall::TMethodCall(const TFunction *func):
fFunc(0), fOffset(0), fClass(0), fMetPtr(0), fDtorOnly(kFALSE), fRetType(kNone)
{
   Init(func);
}
////////////////////////////////////////////////////////////////////////////////
/// Copy ctor.

TMethodCall::TMethodCall(const TMethodCall &orig) :
TObject(orig),
fFunc(orig.fFunc ? gCling->CallFunc_FactoryCopy(orig.fFunc) : 0),
fOffset(orig.fOffset), fClass(orig.fClass), fMetPtr(0 /*!*/),
fMethod(orig.fMethod), fParams(orig.fParams), fProto(orig.fProto),
fDtorOnly(orig.fDtorOnly), fRetType(orig.fRetType)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TMethodCall &TMethodCall::operator=(const TMethodCall &rhs)
{
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

////////////////////////////////////////////////////////////////////////////////
/// TMethodCall dtor.

TMethodCall::~TMethodCall()
{
   gCling->CallFunc_Delete(fFunc);
   delete fMetPtr;
}

////////////////////////////////////////////////////////////////////////////////
/// Return an exact copy of this object.

TObject *TMethodCall::Clone(const char *) const
{
   TObject *newobj = new TMethodCall(*this);
   return newobj;
}

////////////////////////////////////////////////////////////////////////////////
/// Helper function to find the scope associated with a qualified
/// function name

static TClass *R__FindScope(const char *function, UInt_t &pos, ClassInfo_t *cinfo)
{

//______________________________________________________________________________
// FIXME: We don't need to split that into lookup scope and lookup member.
// Consider merging the implementation with the new lookup functionality.

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

////////////////////////////////////////////////////////////////////////////////
/// Initialize the method invocation environment based on
/// the CallFunc object and the TClass describing the function context.

void TMethodCall::Init(TClass *cl, CallFunc_t *function, Longptr_t offset)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Initialize the method invocation environment based on
/// the TFunction object.

void TMethodCall::Init(const TFunction *function)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Initialize the method invocation environment. Necessary input
/// information: the class, method name and the parameter string
/// of the form "\"aap\", 3, 4.35".
///
/// To execute the method call TMethodCall::Execute(object,...).
/// This two step method is much more efficient than calling for
/// every invocation TInterpreter::Execute(...).

void TMethodCall::Init(TClass *cl, const char *method, const char *params, Bool_t objectIsConst /* = kFALSE */)
{
   ClassInfo_t *cinfo = gCling->ClassInfo_Factory();
   if (!cl) {
      UInt_t pos = 0;
      cl = R__FindScope(method,pos,cinfo);
      method = method+pos;
   }
   InitImplementation(method,params,0,objectIsConst,cl,cinfo);
   gCling->ClassInfo_Delete(cinfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the function invocation environment. Necessary input
/// information: the function name and the parameter string of
/// the form "\"aap\", 3, 4.35".
///
/// To execute the method call TMethodCall::Execute(...).
/// This two step method is much more efficient than calling for
/// every invocation TInterpreter::Execute(...).

void TMethodCall::Init(const char *function, const char *params)
{
   UInt_t pos = 0;
   ClassInfo_t *cinfo = gCling->ClassInfo_Factory();
   TClass *cl = R__FindScope(function,pos,cinfo);
   InitImplementation(function+pos, params, 0, false, cl, cinfo);
   gCling->ClassInfo_Delete(cinfo);
}

////////////////////////////////////////////////////////////////////////////////
/// This function implements Init and InitWithPrototype.
///
/// 'methodname' should NOT have any scope information in it.  The scope
/// information should be passed via the TClass or CINT ClassInfo.

void TMethodCall::InitImplementation(const char *methodname, const char *params,
                                     const char *proto,
                                     Bool_t objectIsConst, TClass *cl,
                                     const ClassInfo_t *cinfo,
                                     ROOT::EFunctionMatchMode mode /* = ROOT::kConversionMatch */)
{
   if (!fFunc) {
      R__LOCKGUARD(gInterpreterMutex);
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

   R__LOCKGUARD(gInterpreterMutex);
   if (params && params[0]) {
      gCling->CallFunc_SetFunc(fFunc, scope, (char *)methodname, (char *)params, objectIsConst, &fOffset);
   } else if (proto && proto[0]) {
      gCling->CallFunc_SetFuncProto(fFunc, scope, (char *)methodname, (char *)proto, objectIsConst, &fOffset, mode);
   } else {
      // No parameters
      gCling->CallFunc_SetFuncProto(fFunc, scope, (char *)methodname, "", objectIsConst, &fOffset, mode);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the method invocation environment. Necessary input
/// information: the class, method name and the prototype string of
/// the form: "char*,int,float".
///
/// To execute the method call TMethodCall::Execute(object,...).
/// This two step method is much more efficient than calling for
/// every invocation TInterpreter::Execute(...).

void TMethodCall::InitWithPrototype(TClass *cl, const char *method, const char *proto, Bool_t objectIsConst /* = kFALSE */, ROOT::EFunctionMatchMode mode /* = ROOT::kConversionMatch */)
{
   ClassInfo_t *cinfo = gCling->ClassInfo_Factory();
   if (!cl) {
      UInt_t pos = 0;
      cl = R__FindScope(method,pos,cinfo);
      method = method+pos;
   }
   InitImplementation(method, 0, proto, objectIsConst, cl, cinfo, mode);
   gCling->ClassInfo_Delete(cinfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the function invocation environment. Necessary input
/// information: the function name and the prototype string of
/// the form: "char*,int,float".
///
/// To execute the method call TMethodCall::Execute(...).
/// This two step method is much more efficient than calling for
/// every invocation TInterpreter::Execute(...).

void TMethodCall::InitWithPrototype(const char *function, const char *proto, ROOT::EFunctionMatchMode mode /* = ROOT::kConversionMatch */)
{
   UInt_t pos = 0;
   ClassInfo_t *cinfo = gCling->ClassInfo_Factory();
   TClass *cl = R__FindScope(function,pos,cinfo);
   InitImplementation(function+pos, 0, proto, false, cl, cinfo, mode);
   gCling->ClassInfo_Delete(cinfo);
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if the method call has been properly initialized and is
/// usable.

Bool_t TMethodCall::IsValid() const
{
   return fFunc ? gCling->CallFunc_IsValid(fFunc) : kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the TMethod describing the method to be executed. This takes
/// all overriding and overloading into account (call TClass::GetMethod()).
/// Since finding the method is expensive the result is cached.

TFunction *TMethodCall::GetMethod()
{
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

////////////////////////////////////////////////////////////////////////////////
/// Execute the method (with preset arguments) for the specified object.

void TMethodCall::Execute(void *object)
{
   if (!fFunc) return;

   void *address = 0;
   if (object) address = (void*)((Longptr_t)object + fOffset);
   if (!fDtorOnly && fMethod[0]=='~') {
      Error("Execute","TMethodCall can no longer be use to call the operator delete and the destructor at the same time");
   }
   gCling->CallFunc_Exec(fFunc,address);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the method for the specified object and argument values.

void TMethodCall::Execute(void *object, const char *params)
{
   if (!fFunc) return;

   // SetArgs contains the necessary lock.
   gCling->CallFunc_SetArgs(fFunc, (char *)params);

   void *address = 0;
   if (object) address = (void*)((Longptr_t)object + fOffset);
   gCling->SetTempLevel(1);
   gCling->CallFunc_Exec(fFunc,address);
   gCling->SetTempLevel(-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the method (with preset arguments) for the specified object.

void TMethodCall::Execute(void *object, Longptr_t &retLong)
{
   if (!fFunc) return;

   void *address = 0;
   if (object) address = (void*)((Longptr_t)object + fOffset);
   gCling->SetTempLevel(1);
   retLong = gCling->CallFunc_ExecInt(fFunc,address);
   gCling->SetTempLevel(-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the method for the specified object and argument values.

void TMethodCall::Execute(void *object, const char *params, Longptr_t &retLong)
{
   if (!fFunc) return;

   // SetArgs contains the necessary lock.
   gCling->CallFunc_SetArgs(fFunc, (char *)params);

   void *address = 0;
   if (object) address = (void*)((Longptr_t)object + fOffset);
   gCling->SetTempLevel(1);
   retLong = gCling->CallFunc_ExecInt(fFunc,address);
   gCling->SetTempLevel(-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the method (with preset arguments) for the specified object.

void TMethodCall::Execute(void *object, Double_t &retDouble)
{
   if (!fFunc) return;

   void *address = 0;
   if (object) address = (void*)((Longptr_t)object + fOffset);
   gCling->SetTempLevel(1);
   retDouble = gCling->CallFunc_ExecDouble(fFunc,address);
   gCling->SetTempLevel(-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the method for the specified object and argument values.

void TMethodCall::Execute(void *object, const char *params, Double_t &retDouble)
{
   if (!fFunc) return;

   gCling->CallFunc_SetArgs(fFunc, (char *)params);

   void *address = 0;
   if (object) address = (void*)((Longptr_t)object + fOffset);
   gCling->SetTempLevel(1);
   retDouble = gCling->CallFunc_ExecDouble(fFunc,address);
   gCling->SetTempLevel(-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the method (with preset arguments) for the specified object.

void TMethodCall::Execute(void *object, char **retText)
{
   if (!fFunc) return;

   void *address = 0;
   if (object) address = (void*)((Longptr_t)object + fOffset);
   gCling->SetTempLevel(1);
   *retText =(char*) (gCling->CallFunc_ExecInt(fFunc,address));
   gCling->SetTempLevel(-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the method for the specified object and argument values.

void TMethodCall::Execute(void *object, const char *params, char **retText)
{
   if (!fFunc) return;

   // SetArgs contains the necessary lock.
   gCling->CallFunc_SetArgs(fFunc, (char *)params);

   void *address = 0;
   if (object) address = (void*)((Longptr_t)object + fOffset);
   gCling->SetTempLevel(1);
   *retText =(char*)(gCling->CallFunc_ExecInt(fFunc,address));
   gCling->SetTempLevel(-1);
}

////////////////////////////////////////////////////////////////////////////////
/// Invoke the method
///
/// \param[in] objAddress  Address of the object to execute the method (nullptr if it is a free function)
/// \param[in] args        Array of pointer to the address of the argument to pass to the
///                        function as is.  *No* conversion is done, the argument must be
///                        of the expected type.
/// \param[in] nargs       Number of arguments passed (must be less than actua size of args
/// \param[out] ret        Address of value (or object) to use for the return value.

void TMethodCall::Execute(void *objAddress, const void* args[], int nargs, void *ret /* = 0 */)
{
   if (!fFunc) return;

   gCling->CallFunc_ExecWithArgsAndReturn(fFunc,objAddress,args,nargs,ret);

}

////////////////////////////////////////////////////////////////////////////////
/// Returns the return type of the method. Either (unsigned) long,
/// int, short and char, or float and double or anything else.
/// Since finding the return type is expensive the result is cached.

TMethodCall::EReturnType TMethodCall::ReturnType()
{
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

////////////////////////////////////////////////////////////////////////////////
/// ParamArr is an array containing the function argument values.
/// If nparam = -1 then paramArr must contain values for all function
/// arguments, otherwise Nargs-NargsOpt <= nparam <= Nargs, where
/// Nargs is the number of all arguments and NargsOpt is the number
/// of default arguments.

void TMethodCall::SetParamPtrs(void *paramArr, Int_t nparam)
{
   if (!fFunc) return;
   gCling->CallFunc_SetArgArray(fFunc,(Longptr_t *)paramArr, nparam);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset parameter list. To be used before the first call the SetParam().

void TMethodCall::ResetParam()
{
   if (!fFunc) return;
   gCling->CallFunc_ResetArg(fFunc);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a long method parameter.

void TMethodCall::SetParam(Long_t l)
{
   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,l);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a double method parameter.

void TMethodCall::SetParam(Float_t f)
{
   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,f);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a double method parameter.

void TMethodCall::SetParam(Double_t d)
{
   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,d);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a long long method parameter.

void TMethodCall::SetParam(Long64_t ll)
{
   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,ll);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a unsigned long long method parameter.

void TMethodCall::SetParam(ULong64_t ull)
{
   if (!fFunc) return;
   gCling->CallFunc_SetArg(fFunc,ull);
}
