// @(#)root/net:$Id$
// Author: Fons Rademakers   09/01/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMonitor
#define ROOT_TMonitor


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMonitor                                                             //
//                                                                      //
// This class monitors activity on a number of network sockets.         //
// The actual monitoring is done by TSystem::DispatchOneEvent().        //
// Typical usage: create a TMonitor object. Register a number of        //
// TSocket objects and call TMonitor::Select(). Select() returns the    //
// socket object which has data waiting. TSocket objects can be added,  //
// removed, (temporary) enabled or disabled.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif

class TList;
class TSocket;


class TMonitor : public TObject, public TQObject {

friend class TSocketHandler;
friend class TTimeOutTimer;
friend class TXSlave;
friend class TXSocket;

private:
   TList    *fActive;     //list of sockets to monitor
   TList    *fDeActive;   //list of (temporary) disabled sockets
   TSocket  *fReady;      //socket which is ready to be read or written
   Bool_t    fMainLoop;   //true if monitoring sockets within the main event loop
   Bool_t    fInterrupt;  //flags an interrupt to Select

   void  SetReady(TSocket *sock);
   void *GetSender() { return this; }  // used to get gTQSender

public:
   enum EInterest { kRead = 1, kWrite = 2 };

   TMonitor(Bool_t mainloop = kTRUE);
   TMonitor(const TMonitor &m);
   virtual ~TMonitor();

   virtual void Add(TSocket *sock, Int_t interest = kRead);
   virtual void SetInterest(TSocket *sock, Int_t interest = kRead);
   virtual void Remove(TSocket *sock);
   virtual void RemoveAll();

   virtual void Activate(TSocket *sock);
   virtual void ActivateAll();
   virtual void DeActivate(TSocket *sock);
   virtual void DeActivateAll();
   virtual void Ready(TSocket *sock); // *SIGNAL*

   void     Interrupt() { fInterrupt = kTRUE; }
   void     ResetInterrupt() { fInterrupt = kFALSE; }

   TSocket *Select();
   TSocket *Select(Long_t timeout);
   Int_t    Select(TList *rdready, TList *wrready, Long_t timeout);

   Int_t  GetActive(Long_t timeout = -1) const;
   Int_t  GetDeActive() const;
   TList *GetListOfActives() const;
   TList *GetListOfDeActives() const;

   Bool_t IsActive(TSocket *s) const;

   ClassDef(TMonitor,0)  //Monitor activity on a set of TSocket objects
};

#endif
