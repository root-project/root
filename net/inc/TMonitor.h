// @(#)root/net:$Name:  $:$Id: TMonitor.h,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
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

class TSocket;
class TList;

class TMonitor : public TObject {

private:
   TList    *fActive;     //list of sockets to monitor
   TList    *fDeActive;   //list of (temporary) disabled sockets
   TSocket  *fReady;      //socket which is ready to be read

public:
   TMonitor();
   virtual ~TMonitor();

   void Add(TSocket *sock);
   void Remove(TSocket *sock);
   void RemoveAll();

   void Activate(TSocket *sock);
   void ActivateAll();
   void DeActivate(TSocket *sock);
   void DeActivateAll();

   TSocket *Select();
   TSocket *Select(Long_t timeout);
   void     SetReady(TSocket *sock);

   Int_t  GetActive() const;
   Int_t  GetDeActive() const;

   ClassDef(TMonitor,0)  //Monitor activity on a set of TSocket objects
};

#endif
