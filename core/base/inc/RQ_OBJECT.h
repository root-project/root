// @(#)root/base:$Id$
// Author: Valeriy Onuchin & Fons Rademakers   15/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RQ_OBJECT
#define ROOT_RQ_OBJECT

#include <TQObject.h>
#include <Varargs.h>


//---- RQ_OBJECT macro -----------------------------------------------
//
// Macro is used to delegate TQObject methods to other classes
// Example:
//
//    #include "RQ_OBJECT.h"
//
//    class A {
//       RQ_OBJECT("A")
//    private:
//       Int_t fValue;
//    public:
//       A() : fValue(0) { }
//       ~A() { }
//
//       void  SetValue(Int_t value)
//       void  PrintValue() const { printf("value=%d\n", fValue); }
//       Int_t GetValue() const { return fValue; }
//    };
//
//    void A::SetValue(Int_t value)
//    {
//       // Sets new value
//
//       // to prevent infinite looping in the case
//       // of cyclic connections
//       if (value != fValue) {
//          fValue = value;
//          Emit("SetValue(Int_t)", fValue);
//       }
//    }
//
// Load this class into root session and try the folllowing:
//
// a = new A();
// b = new A();
//
// Here is one way to connect two of these objects together:
//
// a->Connect("SetValue(Int_t)", "A", b, "SetValue(Int_t)");
//
// Calling a->SetValue(79) will make a emit a signal, which b
// will receive, i.e. b->SetValue(79) is invoked. b will in
// turn emit the same signal, which nobody receives, since no
// slot has been connected to it, so it disappears into hyperspace.
//

#define RQ_OBJECT1(sender_class)\
private:\
TQObjSender fQObject;\
public:\
TList *GetListOfSignals() const {return fQObject.GetListOfSignals();}\
Bool_t Connect(const char *sig,const char *cl,void *rcvr,const char *slt)\
{fQObject.SetSender(this);fQObject.SetSenderClassName(sender_class);return fQObject.Connect(sig,cl,rcvr,slt);}\
Bool_t Disconnect(const char *sig=0,void *rcvr=0,const char *slt=0){return fQObject.Disconnect(sig,rcvr,slt);}\
void HighPriority(const char *signal_name,const char *slot_name=0){fQObject.HighPriority(signal_name,slot_name);}\
void LowPriority(const char *signal_name,const char *slot_name=0){fQObject.LowPriority(signal_name,slot_name);}\
template <typename... T> void EmitVA(const char *signal_name, Int_t nargs, const T&... params) \
{ fQObject.EmitVA(signal_name,nargs,params...); } \
void Emit(const char *signal){fQObject.Emit(signal);}\
void Emit(const char *signal,const char *params){fQObject.Emit(signal,params);}\
void Emit(const char *signal,Long_t *paramArr){fQObject.Emit(signal,paramArr);}\
void Emit(const char *signal,Double_t param){fQObject.Emit(signal,param);}\
void Emit(const char *signal,Long_t param){fQObject.Emit(signal,param);}

#define RQ_OBJECT2(sender_class)\
void Emit(const char *signal,Long64_t param){fQObject.Emit(signal,param);}\
void Emit(const char *signal,ULong64_t param){fQObject.Emit(signal,param);}\
void Emit(const char *signal,Bool_t param){Emit(signal,(Long_t)param);}\
void Emit(const char *signal,Char_t param){Emit(signal,(Long_t)param);}\
void Emit(const char *signal,UChar_t param){Emit(signal,(Long_t)param);}\
void Emit(const char *signal,Short_t param){Emit(signal,(Long_t)param);}\
void Emit(const char *signal,UShort_t param){Emit(signal,(Long_t)param);}\
void Emit(const char *signal,Int_t param){Emit(signal,(Long_t)param);}\
void Emit(const char *signal,UInt_t param){Emit(signal,(Long_t)param);}\
void Emit(const char *signal,ULong_t param){Emit(signal,(Long_t)param);}\
void Emit(const char *signal,Float_t param){Emit(signal,(Double_t)param);}\
void Destroyed(){Emit("Destroyed()");}\
void ChangedBy(const char *method){Emit("ChangedBy(char*)",method);}\
void Message(const char *msg){Emit("Message(char*)",msg);}\
private:

#define RQ_OBJECT(sender_class)\
   RQ_OBJECT1(sender_class)\
   RQ_OBJECT2(sender_class)

#endif
