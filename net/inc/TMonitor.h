// @(#)root/net:$Name:  $:$Id: TMonitor.h,v 1.3 2001/01/25 18:39:42 rdm Exp $
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

class TList;
class TSocket;


class TMonitor : public TObject {

friend class TSocketHandler;
friend class TTimeOutTimer;

private:
   TList    *fActive;     //list of sockets to monitor
   TList    *fDeActive;   //list of (temporary) disabled sockets
   TSocket  *fReady;      //socket which is ready to be read or written

   void SetReady(TSocket *sock);

public:
   enum EInterest { kRead = 1, kWrite = 2 };

   TMonitor();
   virtual ~TMonitor();

   void Add(TSocket *sock, EInterest interest = kRead);
   void Remove(TSocket *sock);
   void RemoveAll();

   void Activate(TSocket *sock);
   void ActivateAll();
   void DeActivate(TSocket *sock);
   void DeActivateAll();

   TSocket *Select();
   TSocket *Select(Long_t timeout);

   Int_t        GetActive() const;
   Int_t        GetDeActive() const;
   const TList *GetListOfActives() const { return fActive; }
   const TList *GetListOfDeActives() const { return fDeActive; }

   ClassDef(TMonitor,0)  //Monitor activity on a set of TSocket objects
};

#endif
