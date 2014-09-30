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

#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif
#ifndef ROOT_Varargs
#include "Varargs.h"
#endif
#ifndef ROOT_TInterpreter
#include "TInterpreter.h"
#endif

class TQSlot;


class TQConnection : public TList, public TQObject {

protected:
   TQSlot  *fSlot;       // slot-method calling interface
   void    *fReceiver;   // ptr to object to which slot is applied
   TString  fClassName;  // class name of the receiver

   virtual void PrintCollectionHeader(Option_t* option) const;

   Bool_t      CheckSlot(Int_t nargs) const;
   void       *GetSlotAddress() const;
   CallFunc_t *LockSlot() const;
   void        UnLockSlot(TQSlot *) const;

   TQConnection &operator=(const TQConnection &) = delete;

public:
   TQConnection();
   TQConnection(TClass* cl, void *receiver, const char *method_name);
   TQConnection(const char *class_name, void *receiver,
                const char *method_name);
   TQConnection(const TQConnection &con);
   virtual ~TQConnection();

   const char *GetName() const;
   void *GetReceiver() const { return fReceiver; }
   const char *GetClassName() const { return fClassName; }
   void Destroyed();         // *SIGNAL*
   void ExecuteMethod();

   void ExecuteMethod(Int_t nargs, va_list va) = delete;

   template <typename... T> inline void ExecuteMethod(const T&... params)
   {
      if (!CheckSlot(sizeof...(params))) return;

      CallFunc_t *func = LockSlot();

      void *address = GetSlotAddress();
      TQSlot *s = fSlot;

      gInterpreter->CallFunc_SetArguments(func,params...);
      gInterpreter->CallFunc_Exec(func, address);

      UnLockSlot(s);
   }

   template <typename... T> inline void ExecuteMethod(Int_t /* nargs */, const T&... params)
   {
      ExecuteMethod(params...);
   }

   //void ExecuteMethod(Int_t nargs, va_list va);
   void ExecuteMethod(Long_t param);
   void ExecuteMethod(Long64_t param);
   void ExecuteMethod(Double_t param);
   void ExecuteMethod(Long_t *params, Int_t nparam = -1);
   void ExecuteMethod(const char *params);
   void ls(Option_t *option="") const;

   ClassDef(TQConnection,0) // Internal class used in the object communication mechanism
};

R__EXTERN char *gTQSlotParams; // used to pass string parameters

#ifndef ROOT_TQObjectEmitVA
#include "TQObjectEmitVA.h"
#endif

#endif
