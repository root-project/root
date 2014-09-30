// @(#)root/base:$Id$
// Author: Valeriy Onuchin & Fons Rademakers   15/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQConnection class is an internal class, used in the object          //
// communication mechanism.                                             //
//                                                                      //
// TQConnection:                                                        //
//    -  is a list of signal_lists containing pointers                  //
//       to this connection                                             //
//    -  receiver is the object to which slot-method is applied         //
//                                                                      //
// This implementation is provided by                                   //
// Valeriy Onuchin (onuchin@sirius.ihep.su).                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Varargs.h"
#include "TQConnection.h"
#include "TROOT.h"
#include "TRefCnt.h"
#include "TClass.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TDataType.h"
#include "TInterpreter.h"
#include "Riostream.h"
#include "TVirtualMutex.h"
#include "THashTable.h"

ClassImpQ(TQConnection)

char *gTQSlotParams; // used to pass string parameter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TQSlot = slightly modified TMethodCall class                        //
//           used in the object communication mechanism                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TQSlot : public TObject, public TRefCnt {

protected:
   CallFunc_t    *fFunc;      // CINT method invocation environment
   ClassInfo_t   *fClass;     // CINT class for fFunc
   TFunction     *fMethod;    // slot method or global function
   Long_t         fOffset;    // offset added to object pointer
   TString        fName;      // full name of method
   Int_t          fExecuting; // true if one of this slot's ExecuteMethod methods is being called
public:
   TQSlot(TClass *cl, const char *method, const char *funcname);
   TQSlot(const char *class_name, const char *funcname);
   virtual ~TQSlot();

   Bool_t      CheckSlot(Int_t nargs) const;
   Long_t      GetOffset() const { return fOffset; }
   CallFunc_t *StartExecuting();
   void        EndExecuting();

   const char *GetName() const {
      return fName.Data();
   }

   void ExecuteMethod(void *object, Int_t nargs, va_list ap) = delete;
   void ExecuteMethod(void *object);
   void ExecuteMethod(void *object, Long_t param);
   void ExecuteMethod(void *object, Long64_t param);
   void ExecuteMethod(void *object, Double_t param);
   void ExecuteMethod(void *object, const char *params);
   void ExecuteMethod(void *object, Long_t *paramArr, Int_t nparam = -1);
   void Print(Option_t *opt = "") const;
   void ls(Option_t *opt = "") const {
      Print(opt);
   }

   Bool_t IsExecuting() const {
      return fExecuting > 0;
   }
};


//______________________________________________________________________________
TQSlot::TQSlot(TClass *cl, const char *method_name,
               const char *funcname) : TObject(), TRefCnt()
{
   // Create the method invocation environment. Necessary input
   // information: the class, full method name with prototype
   // string of the form: method(char*,int,float).
   // To initialize class method with default arguments, method
   // string with default parameters should be of the form:
   // method(=\"ABC\",1234,3.14) (!! parameter string should
   // consists of '=').
   // To execute the method call TQSlot::ExecuteMethod(object,...).

   fFunc      = 0;
   fClass     = 0;
   fOffset    = 0;
   fMethod    = 0;
   fName      = "";
   fExecuting = 0;

   // cl==0, is the case of interpreted function.

   fName = method_name;

   char *method = new char[strlen(method_name) + 1];
   if (method) strcpy(method, method_name);

   char *proto;
   char *tmp;
   char *params = 0;

   // separate method and protoype strings

   if ((proto = strchr(method, '('))) {

      // substitute first '(' symbol with '\0'
      *proto++ = '\0';

      // last ')' symbol with '\0'
      if ((tmp = strrchr(proto, ')'))) * tmp = '\0';
      if ((params = strchr(proto, '='))) * params = ' ';
   }

   R__LOCKGUARD2(gInterpreterMutex);
   fFunc = gCling->CallFunc_Factory();

   // initiate class method (function) with proto
   // or with default params

   if (cl) {
      if (params) {
         gCling->CallFunc_SetFunc(fFunc, cl->GetClassInfo(), method, params, &fOffset);
         fMethod = cl->GetMethod(method, params);
      } else {
         gCling->CallFunc_SetFuncProto(fFunc, cl->GetClassInfo(), method, proto, &fOffset);
         fMethod = cl->GetMethodWithPrototype(method, proto);
      }
   } else {
      fClass = gCling->ClassInfo_Factory();
      if (params) {
         gCling->CallFunc_SetFunc(fFunc, fClass, (char *)funcname, params, &fOffset);
         fMethod = gROOT->GetGlobalFunction(funcname, params, kFALSE);
      } else {
         gCling->CallFunc_SetFuncProto(fFunc, fClass, (char *)funcname, proto, &fOffset);
         fMethod = gROOT->GetGlobalFunctionWithPrototype(funcname, proto, kFALSE);
      }
   }

   // cleaning
   delete [] method;
}

//______________________________________________________________________________
TQSlot::TQSlot(const char *class_name, const char *funcname) :
   TObject(), TRefCnt()
{
   // Create the method invocation environment. Necessary input
   // information: the name of class (could be interpreted class),
   // full method name with prototype or parameter string
   // of the form: method(char*,int,float).
   // To initialize class method with default arguments, method
   // string with default parameters  should be of the form:
   // method(=\"ABC\",1234,3.14) (!! parameter string should
   // consists of '=').
   // To execute the method call TQSlot::ExecuteMethod(object,...).

   fFunc      = 0;
   fClass     = 0;
   fOffset    = 0;
   fMethod    = 0;
   fName      = funcname;
   fExecuting = 0;

   char *method = new char[strlen(funcname) + 1];
   if (method) strcpy(method, funcname);

   char *proto;
   char *tmp;
   char *params = 0;

   // separate method and protoype strings

   if ((proto =  strchr(method, '('))) {
      *proto++ = '\0';
      if ((tmp = strrchr(proto, ')'))) * tmp  = '\0';
      if ((params = strchr(proto, '='))) * params = ' ';
   }

   R__LOCKGUARD2(gInterpreterMutex);
   fFunc = gCling->CallFunc_Factory();
   gCling->CallFunc_IgnoreExtraArgs(fFunc, true);

   fClass = gCling->ClassInfo_Factory();
   TClass *cl = 0;

   if (!class_name)
      ;                       // function
   else {
      gCling->ClassInfo_Init(fClass, class_name);  // class
      cl = TClass::GetClass(class_name);
   }

   if (params) {
      gCling->CallFunc_SetFunc(fFunc, fClass, method, params, &fOffset);
      if (cl)
         fMethod = cl->GetMethod(method, params);
      else
         fMethod = gROOT->GetGlobalFunction(method, params, kTRUE);
   } else {
      gCling->CallFunc_SetFuncProto(fFunc, fClass, method, proto , &fOffset);
      if (cl)
         fMethod = cl->GetMethodWithPrototype(method, proto);
      else
         fMethod = gROOT->GetGlobalFunctionWithPrototype(method, proto, kTRUE);
   }

   delete [] method;
}

//______________________________________________________________________________
TQSlot::~TQSlot()
{
   // TQSlot dtor.

   // don't delete executing environment of a slot that is being executed
   if (!fExecuting) {
      gCling->CallFunc_Delete(fFunc);
      gCling->ClassInfo_Delete(fClass);
   }
}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object)
{
   // ExecuteMethod the method (with preset arguments) for
   // the specified object.

   ExecuteMethod(object, (Long_t*)nullptr, 0);

}

//______________________________________________________________________________
inline Bool_t TQSlot::CheckSlot(Int_t nargs) const
{
   // Return true if the method is valid and the number of arguments is
   // acceptable.

   if (!fMethod) {
      Error("ExecuteMethod", "method %s not found,"
            "\n(note: interpreted methods are not supported with varargs)",
            fName.Data());
      return kFALSE;
   }

   if (nargs < fMethod->GetNargs() - fMethod->GetNargsOpt() ||
       nargs > fMethod->GetNargs()) {
      Error("ExecuteMethod", "nargs (%d) not consistent with expected number of arguments ([%d-%d])",
            nargs, fMethod->GetNargs() - fMethod->GetNargsOpt(),
            fMethod->GetNargs());
      return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
CallFunc_t *TQSlot::StartExecuting() {
   // Mark the slot as executing.

   fExecuting++;
   return fFunc;
}

//______________________________________________________________________________
void TQSlot::EndExecuting() {
   // Mark the slot as no longer executing and cleanup if need be.

   fExecuting--;
   if (!TestBit(kNotDeleted) && !fExecuting)
      gCling->CallFunc_Delete(fFunc);
}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, Long_t param)
{
   // ExecuteMethod the method for the specified object and
   // with single argument value.

   ExecuteMethod(object, &param, 1);

}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, Long64_t param)
{
   // ExecuteMethod the method for the specified object and
   // with single argument value.

   Long_t *arg = reinterpret_cast<Long_t *>(&param);
   ExecuteMethod(object, arg, 1);

}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, Double_t param)
{
   // ExecuteMethod the method for the specified object and
   // with single argument value.

   Long_t *arg = reinterpret_cast<Long_t *>(&param);
   ExecuteMethod(object, arg, 1);

}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, const char *param)
{
   // ExecuteMethod the method for the specified object and text param.

   Long_t arg = reinterpret_cast<Long_t>(param);
   ExecuteMethod(object, &arg, 1);

}

//______________________________________________________________________________
inline void TQSlot::ExecuteMethod(void *object, Long_t *paramArr, Int_t nparam)
{
   // ExecuteMethod the method for the specified object and with
   // several argument values.
   // ParamArr is an array containing the function argument values.
   // If nparam = -1 then paramArr must contain values for all function
   // arguments, otherwise Nargs-NargsOpt <= nparam <= Nargs, where
   // Nargs is the number of all arguments and NargsOpt is the number
   // of default arguments.

   void *address = 0;
   R__LOCKGUARD2(gInterpreterMutex);
   if (paramArr) gCling->CallFunc_SetArgArray(fFunc, paramArr, nparam);
   if (object) address = (void *)((Long_t)object + fOffset);
   fExecuting++;
   gCling->CallFunc_Exec(fFunc, address);
   fExecuting--;
   if (!TestBit(kNotDeleted) && !fExecuting)
      gCling->CallFunc_Delete(fFunc);
}

//______________________________________________________________________________
void TQSlot::Print(Option_t *) const
{
   // Print info about slot.

   std::cout << IsA()->GetName() << "\t" << GetName() << "\t"
             << "Number of Connections = " << References() << std::endl;
}

////////////////////////////////////////////////////////////////////////////////

class TQSlotPool {
private:
   THashTable *fTable;
public:
   TQSlotPool() {
      fTable = new THashTable(50);
   }
   virtual ~TQSlotPool() {
      fTable->Clear("nodelete");
   }

   TQSlot  *New(const char *class_name, const char *funcname);
   TQSlot  *New(TClass *cl, const char *method, const char *func);
   void     Free(TQSlot *slot);
};

//______________________________________________________________________________
TQSlot *TQSlotPool::New(const char *class_name, const char *funcname)
{
   // Create new slot or return already existing one.

   TString name = class_name;
   name += "::";
   name += funcname;

   TQSlot *slot = (TQSlot *)fTable->FindObject(name.Data());

   if (!slot) {
      slot = new TQSlot(class_name, funcname);
      fTable->Add(slot);
   }
   slot->AddReference();
   return slot;
}

//______________________________________________________________________________
TQSlot *TQSlotPool::New(TClass *cl, const char *method, const char *func)
{
   // Create new slot or return already existing one.

   TString name;

   if (cl) {
      name = cl->GetName();
      name += "::";
      name += method;
   } else {
      name = "::";
      name += func;
   }

   TQSlot *slot = (TQSlot *)fTable->FindObject(name.Data());

   if (!slot) {
      slot = new TQSlot(cl, method, func);
      fTable->Add(slot);
   }
   slot->AddReference();
   return slot;
}

//______________________________________________________________________________
void TQSlotPool::Free(TQSlot *slot)
{
   // Delete slot if there is no reference to it.

   slot->RemoveReference();  // decrease references to slot

   if (slot->References() <= 0) {
      fTable->Remove(slot);
      if (!slot->IsExecuting()) SafeDelete(slot);
   }
}

static TQSlotPool gSlotPool;  // global pool of slots

////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TQConnection::TQConnection() : TList(), TQObject()
{
   // Default constructor.

   fReceiver = 0;
   fSlot     = 0;
}

//______________________________________________________________________________
TQConnection::TQConnection(TClass *cl, void *receiver, const char *method_name)
   : TList(), TQObject()
{
   // TQConnection ctor.
   //    cl != 0  - connection to object == receiver of class == cl
   //               and method == method_name
   //    cl == 0  - connection to function with name == method_name

   const char *funcname = 0;
   fReceiver = receiver;      // fReceiver is pointer to receiver

   if (!cl) {
      funcname = gCling->Getp2f2funcname(fReceiver);
      if (!funcname)
         Warning("TQConnection", "%s cannot be compiled", method_name);
   }

   if (cl) fClassName = cl->GetName();
   fSlot = gSlotPool.New(cl, method_name, funcname);
}

//______________________________________________________________________________
TQConnection::TQConnection(const char *class_name, void *receiver,
                           const char *funcname) : TList(), TQObject()
{
   // TQConnection ctor.
   //    Creates connection to method of class specified by name,
   //    it could be interpreted class and with method == funcname.

   fClassName = class_name;
   fSlot = gSlotPool.New(class_name, funcname);  // new slot-method
   fReceiver = receiver;      // fReceiver is pointer to receiver
}

//______________________________________________________________________________
TQConnection::TQConnection(const TQConnection &con): TList(), TQObject()
{
   // Copy constructor. Ignore connections to this TQConnections

   fClassName = con.fClassName;
   fSlot = con.fSlot;
   fSlot->AddReference();
   fReceiver = con.fReceiver;
}

//______________________________________________________________________________
TQConnection::~TQConnection()
{
   // TQConnection dtor.
   //    - remove this connection from all signal lists
   //    - we do not delete fSlot if it has other connections,
   //      TQSlot::fCounter > 0 .

   TIter next(this);
   TList *list;

   while ((list = (TList *)next())) {
      list->Remove(this);
      if (list->IsEmpty()) delete list;   // delete empty list
   }
   Clear("nodelete");

   if (!fSlot) return;
   gSlotPool.Free(fSlot);
}

//______________________________________________________________________________
const char *TQConnection::GetName() const
{
   // Returns name of connection (aka name of slot)

   return fSlot->GetName();
}

//______________________________________________________________________________
void TQConnection::Destroyed()
{
   // Signal Destroyed tells that connection is destroyed.

   MakeZombie();
   Emit("Destroyed()");
}

//______________________________________________________________________________
void TQConnection::ls(Option_t *option) const
{
   // List TQConnection full method name and list all signals
   // connected to this connection.

   std::cout << "\t" <<  IsA()->GetName() << "\t" << GetName() << std::endl;
   ((TQConnection *)this)->R__FOR_EACH(TList, ls)(option);
}

//______________________________________________________________________________
void TQConnection::PrintCollectionHeader(Option_t *) const
{
   // Print TQConnection full method name and print all
   // signals connected to this connection.

   TROOT::IndentLevel();
   std::cout << IsA()->GetName() << "\t" << fReceiver << "\t" << GetName() << std::endl;
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod()
{
   // Apply slot-method to the fReceiver object without arguments.

   // This connection might be deleted in result of the method execution
   // (for example in case of a Disconnect).  Hence we do not assume
   // the object is still valid on return.
   TQSlot *s = fSlot;
   fSlot->ExecuteMethod(fReceiver);
   if (s->References() <= 0) delete s;
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(Long_t param)
{
   // Apply slot-method to the fReceiver object with
   // single argument value.

   // This connection might be deleted in result of the method execution
   // (for example in case of a Disconnect).  Hence we do not assume
   // the object is still valid on return.
   TQSlot *s = fSlot;
   fSlot->ExecuteMethod(fReceiver, param);
   if (s->References() <= 0) delete s;
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(Long64_t param)
{
   // Apply slot-method to the fReceiver object with
   // single argument value.

   // This connection might be deleted in result of the method execution
   // (for example in case of a Disconnect).  Hence we do not assume
   // the object is still valid on return.
   TQSlot *s = fSlot;
   fSlot->ExecuteMethod(fReceiver, param);
   if (s->References() <= 0) delete s;
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(Double_t param)
{
   // Apply slot-method to the fReceiver object with
   // single argument value.

   // This connection might be deleted in result of the method execution
   // (for example in case of a Disconnect).  Hence we do not assume
   // the object is still valid on return.
   TQSlot *s = fSlot;
   fSlot->ExecuteMethod(fReceiver, param);
   if (s->References() <= 0) delete s;
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(Long_t *params, Int_t nparam)
{
   // Apply slot-method to the fReceiver object with variable
   // number of argument values.

   // This connection might be deleted in result of the method execution
   // (for example in case of a Disconnect).  Hence we do not assume
   // the object is still valid on return.
   TQSlot *s = fSlot;
   fSlot->ExecuteMethod(fReceiver, params, nparam);
   if (s->References() <= 0) delete s;
}

//______________________________________________________________________________
void TQConnection::ExecuteMethod(const char *param)
{
   // Apply slot-method to the fReceiver object and
   // with string parameter.

   // This connection might be deleted in result of the method execution
   // (for example in case of a Disconnect).  Hence we do not assume
   // the object is still valid on return.
   TQSlot *s = fSlot;
   fSlot->ExecuteMethod(fReceiver, param);
   if (s->References() <= 0) delete s;
}

//______________________________________________________________________________
Bool_t TQConnection::CheckSlot(Int_t nargs) const {
   // Return true if the underlying method is value and the number of argument
   // is compatible.

   return fSlot->CheckSlot(nargs);
}

//______________________________________________________________________________
void *TQConnection::GetSlotAddress() const {
   // Return the object address to be passed to the function.

   if (fReceiver) return (void *)((Long_t)fReceiver + fSlot->GetOffset());
   else return nullptr;
}

//______________________________________________________________________________
CallFunc_t *TQConnection::LockSlot() const {
   // Lock the interpreter and mark the slot as executing.

   if (gInterpreterMutex) gInterpreterMutex->Lock();
   return fSlot->StartExecuting();
}

//______________________________________________________________________________
void TQConnection::UnLockSlot(TQSlot *s) const {
   // Unlock the interpreter and mark the slot as no longer executing.

   s->EndExecuting();
   if (s->References() <= 0) delete s;
   if (gInterpreterMutex) gInterpreterMutex->UnLock();
}
