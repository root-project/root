// @(#)root/meta:$Name:  $:$Id: TMethodCall.cxx,v 1.3 2000/10/31 11:18:25 brun Exp $
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

#include "TMethodCall.h"
#include "TMethod.h"
#include "TClass.h"
#include "TROOT.h"
#include "Strlen.h"
#include "Api.h"

#include "G__ci.h"

#ifndef WIN32
extern long G__globalvarpointer;
#endif


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
   fRetType  = (EReturnType) -1;
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
TMethodCall::~TMethodCall()
{
   // TMethodCall dtor.

   delete fFunc;
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

   if (!fFunc)
      fFunc = new G__CallFunc;
   else
      fFunc->Init();

   fClass    = cl;
   fMetPtr   = 0;
   fMethod   = method;
   fParams   = params;
   fProto    = "";
   fDtorOnly = kFALSE;
   fRetType  = (EReturnType) -1;

   if (cl)
      fFunc->SetFunc(cl->GetClassInfo(), (char *)method, (char *)params, &fOffset);
   else {
      G__ClassInfo gcl;
      fFunc->SetFunc(&gcl, (char *)method, (char *)params, &fOffset);
   }
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

   Init(0, function, params);
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

   if (!fFunc)
      fFunc = new G__CallFunc;
   else
      fFunc->Init();

   fClass    = cl;
   fMetPtr   = 0;
   fMethod   = method;
   fParams   = "";
   fProto    = proto;
   fDtorOnly = kFALSE;
   fRetType  = (EReturnType) -1;

   if (cl)
      fFunc->SetFuncProto(cl->GetClassInfo(), (char *)method, (char *)proto, &fOffset);
   else {
      G__ClassInfo gcl;
      fFunc->SetFuncProto(&gcl, (char *)method, (char *)proto, &fOffset);
   }
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

   InitWithPrototype(0, function, proto);
}

//______________________________________________________________________________
TFunction *TMethodCall::GetMethod()
{
   // Returns the TMethod describing the method to be executed. This takes
   // all overriding and overloading into account (call TClass::GetMethod()).
   // Since finding the method is expensive the result is cached.

   if (!fMetPtr) {
      if (fClass) {
         if (fProto == "")
            fMetPtr = fClass->GetMethod(fMethod.Data(), fParams.Data());
         else
            fMetPtr = fClass->GetMethodWithPrototype(fMethod.Data(), fProto.Data());
      } else {
         if (fProto == "")
            fMetPtr = gROOT->GetGlobalFunction(fMethod.Data(), fParams.Data(), kTRUE);
         else
            fMetPtr = gROOT->GetGlobalFunctionWithPrototype(fMethod.Data(), fProto.Data(), kTRUE);
      }
   }
   return fMetPtr;
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object)
{
   // Execute the method (with preset arguments) for the specified object.

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   if (fDtorOnly) {
#ifdef WIN32
      long saveglobalvar = G__getgvp();
      G__setgvp((long)address);
      fFunc->Exec(address);
      G__setgvp(saveglobalvar);
#else
      long saveglobalvar = G__globalvarpointer;
      G__globalvarpointer = (long)address;
      fFunc->Exec(address);
      G__globalvarpointer = saveglobalvar;
#endif
   } else
      fFunc->Exec(address);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params)
{
   // Execute the method for the specified object and argument values.

   fFunc->SetArgs((char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   fFunc->Exec(address);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, Long_t &retLong)
{
   // Execute the method (with preset arguments) for the specified object.

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   retLong = fFunc->ExecInt(address);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, Long_t &retLong)
{
   // Execute the method for the specified object and argument values.

   fFunc->SetArgs((char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   retLong = fFunc->ExecInt(address);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, Double_t &retDouble)
{
   // Execute the method (with preset arguments) for the specified object.

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   retDouble = fFunc->ExecDouble(address);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, Double_t &retDouble)
{
   // Execute the method for the specified object and argument values.

   fFunc->SetArgs((char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   retDouble = fFunc->ExecDouble(address);
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, char **retText)
{
   // Execute the method (with preset arguments) for the specified object.

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   *retText =(char*) (fFunc->ExecInt(address));
}

//______________________________________________________________________________
void TMethodCall::Execute(void *object, const char *params, char **retText)
{
   // Execute the method for the specified object and argument values.

   fFunc->SetArgs((char *)params);

   void *address = 0;
   if (object) address = (void*)((Long_t)object + fOffset);
   *retText =(char*)( fFunc->ExecInt(address));
}

//______________________________________________________________________________
TMethodCall::EReturnType TMethodCall::ReturnType()
{
   // Returns the return type of the method. Either (unsigned) long,
   // int, short and char, or float and double or anything else.
   // Since finding the return type is expensive the result is cached.

   if ((int)fRetType == -1) {
      TFunction *func = GetMethod();
      if (func == 0) {
         fRetType = kOther;
         Error("ReturnType","Unknown method");
         return kOther;
      }
      G__TypedefInfo type(func->GetReturnTypeName());
      const char *name = type.TrueName();

      if (!strcmp("(unknown)",name)) {
         G__TypeInfo type(func->GetReturnTypeName());
         name = type.TrueName();
      }

      if (!strcmp("unsigned int", name)   || !strcmp("int", name)     ||
          !strcmp("unsigned long", name)  || !strcmp("long", name)    ||
          !strcmp("unsigned short", name) || !strcmp("short", name)   ||
          !strcmp("unsigned char", name)  || !strcmp("char", name)    ||
          !strcmp("UInt_t", name)         || !strcmp("Int_t", name)   ||
          !strcmp("ULong_t", name)        || !strcmp("Long_t", name)  ||
          !strcmp("UShort_t", name)       || !strcmp("Short_t", name) ||
          !strcmp("UChar_t", name)        || !strcmp("Char_t", name)  ||
          strstr(name, "enum"))
         fRetType = kLong;
      else if (!strcmp("float", name)   || !strcmp("double", name)    ||
               !strcmp("Float_t", name) || !strcmp("Double_t", name))
         fRetType = kDouble;
      else
         fRetType = kOther;
   }
   return fRetType;
}

//______________________________________________________________________________
void TMethodCall::SetParamPtrs(void *paramArr)
{
   // ParamArr is an array containing the addresses where to take the
   // function parameters. At least as many pointers should be present in
   // the array as there are required arguments (all arguments - default args).

   fFunc->SetArgArray((long *)paramArr);
}
