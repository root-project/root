// @(#)root/base:$Id$
// Author: Valeriy Onuchin & Fons Rademakers   15/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQConnection
#define ROOT_TQConnection

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

#include "TInterpreter.h"
#include "TQObject.h"
#include "TVirtualQConnection.h"

class TQSlot;


class TQConnection : public TVirtualQConnection, public TQObject {
protected:
   TQSlot  *fSlot = 0;       // slot-method calling interface
   void    *fReceiver = 0;   // ptr to object to which slot is applied
   TString  fClassName;  // class name of the receiver

   virtual void PrintCollectionHeader(Option_t* option) const override;

   Bool_t      CheckSlot(Int_t nargs) const;
   void       *GetSlotAddress() const;
   CallFunc_t *LockSlot() const;
   void        UnLockSlot(TQSlot *) const;
   virtual CallFunc_t *GetSlotCallFunc() const override;

   TQConnection &operator=(const TQConnection &) = delete;

   virtual void SetArg(Long_t param) override { SetArgImpl(param); }
   virtual void SetArg(ULong_t param) override { SetArgImpl(param); }
   virtual void SetArg(Float_t param) override { SetArgImpl(param); }
   virtual void SetArg(Double_t param) override { SetArgImpl(param); }
   virtual void SetArg(Long64_t param) override { SetArgImpl(param); }
   virtual void SetArg(ULong64_t param) override { SetArgImpl(param); }
   virtual void SetArg(const char * param) override { SetArgImpl(param); }

   virtual void SetArg(const Longptr_t *params, Int_t nparam = -1) override;

   template <typename T> void SetArgImpl(T arg)
   {
      CallFunc_t *func = GetSlotCallFunc();
      gInterpreter->CallFunc_SetArg(func, arg);
   }

   virtual void SendSignal() override
   {
      CallFunc_t *func = LockSlot();

      void *address = GetSlotAddress();
      TQSlot *s = fSlot;

      gInterpreter->CallFunc_Exec(func, address);

      UnLockSlot(s);
   };

public:
   TQConnection() {}
   TQConnection(TClass* cl, void *receiver, const char *method_name);
   TQConnection(const char *class_name, void *receiver,
                const char *method_name);
   TQConnection(const TQConnection &con);
   virtual ~TQConnection();

   const char *GetName() const override;
   void *GetReceiver() const { return fReceiver; }
   const char *GetClassName() const { return fClassName; }
   void Destroyed() override;         // *SIGNAL*

   void ExecuteMethod(Int_t nargs, va_list va) = delete;
   template <typename... T> inline void ExecuteMethod(const T&... params)
   {
      if (!CheckSlot(sizeof...(params))) return;
      SetArgs(params...);
      SendSignal();
   }

   template <typename... T> inline void ExecuteMethod(Int_t /* nargs */, const T&... params)
   {
      ExecuteMethod(params...);
   }

   // FIXME: Remove and fallback to the variadic template.
   // FIXME: Remove duplication of code in SendSignal and ExecuteMethod overloads.
   void ExecuteMethod();
   void ExecuteMethod(Long_t param);
   void ExecuteMethod(Long64_t param);
   void ExecuteMethod(Double_t param);
   void ExecuteMethod(Longptr_t *params, Int_t nparam = -1);
   void ExecuteMethod(const char *params);
   void ls(Option_t *option="") const override;

   ClassDefOverride(TQConnection,0) // Internal class used in the object communication mechanism
};

R__EXTERN char *gTQSlotParams; // used to pass string parameters

#endif
