// @(#)root/base:$Name:$:$Id:$
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

#ifndef ROOT_Tlist
#include "TList.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif


class TQSlot;


class TQConnection : public TList, public TQObject {

protected:
   TQSlot  *fSlot;      // slot-method calling interface
   void    *fReceiver;  // ptr to object to which slot is applied

public:
   TQConnection();
   virtual ~TQConnection();

   TQConnection(TClass* cl, void *receiver, const char *method_name);
   TQConnection(const char *class_name, void *receiver,
                const char *method_name);

   const char *GetName() const;
   void *GetReceiver() const { return fReceiver; }
   void Destroyed();         // *SIGNAL*
   void ExecuteMethod();
   void ExecuteMethod(Long_t param);
   void ExecuteMethod(Double_t param);
   void ExecuteMethod(Long_t *params);
   void ExecuteMethod(const char *params);
   void ls(Option_t *option="");
   void Print(Option_t *option="");

   ClassDef(TQConnection,0) // Internal class used in the object communication mechanism
};

R__EXTERN char *gTQSlotParams; // used to pass string parameters

#endif
