// @(#)root/net:$Name:  $:$Id: TMonitor.cxx,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Fons Rademakers   09/01/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TMonitor.h"
#include "TSocket.h"
#include "TList.h"
#include "TSystem.h"
#include "TSysEvtHandler.h"
#include "TTimer.h"
#include "TError.h"


//---- Socket event handler ----------------------------------------------------
//
// This utility class is only used via TMonitor.
//

class TSocketHandler : public TFileHandler {
private:
   TMonitor  *fMonitor;   //monitor to which this handler belongs
   TSocket   *fSocket;    //socket being handled

public:
   TSocketHandler(TMonitor *m, TSocket *s);
   Bool_t   Notify();
   Bool_t   ReadNotify() { return Notify(); }
   TSocket *GetSocket() const { return fSocket; }
};

TSocketHandler::TSocketHandler(TMonitor *m, TSocket *s)
               : TFileHandler(s->GetDescriptor(), 1)
{
   fMonitor = m;
   fSocket  = s;

   gSystem->AddFileHandler(this);
}

Bool_t TSocketHandler::Notify()
{
   fMonitor->SetReady(fSocket);
   return kTRUE;
}

//---- Timeout timer -----------------------------------------------------------
//
// This utility class is only used via TMonitor::Select(Int_t timeout)
//

class TTimeOutTimer : public TTimer {
private:
   TMonitor   *fMonitor;   //monitor to which this timer belongs

public:
   TTimeOutTimer(TMonitor *m, Long_t ms);
   Bool_t  Notify();
};

TTimeOutTimer::TTimeOutTimer(TMonitor *m, Long_t ms)
              : TTimer(ms, kTRUE)
{
   fMonitor = m;
   gSystem->AddTimer(this);
}

Bool_t TTimeOutTimer::Notify()
{
   fMonitor->SetReady((TSocket *)-1);
   Remove();       // one shot only
   return kTRUE;
}
//------------------------------------------------------------------------------


ClassImp(TMonitor)

//______________________________________________________________________________
TMonitor::TMonitor()
{
   // Create a monitor object.

   Assert(gSystem);

   fActive   = new TList;
   fDeActive = new TList;
}

//______________________________________________________________________________
TMonitor::~TMonitor()
{
   // Cleanup the monitor object. Does not delete socket being monitored.

   fActive->Delete();
   SafeDelete(fActive);

   fDeActive->Delete();
   SafeDelete(fDeActive);
}

//______________________________________________________________________________
void TMonitor::Add(TSocket *sock)
{
   // Add socket to the monitor's active list.

   fActive->Add(new TSocketHandler(this, sock));
}

//______________________________________________________________________________
void TMonitor::Remove(TSocket *sock)
{
   // Remove a socket from the monitor.

   TIter next(fActive);
   TSocketHandler *s;

   while ((s = (TSocketHandler *) next())) {
      if (sock == s->GetSocket()) {
         fActive->Remove(s);
         delete s;
         return;
      }
   }

   TIter next1(fDeActive);

   while ((s = (TSocketHandler *) next1())) {
      if (sock == s->GetSocket()) {
         fDeActive->Remove(s);
         delete s;
         return;
      }
   }
}

//______________________________________________________________________________
void TMonitor::RemoveAll()
{
   // Remove all sockets from the monitor.

   fActive->Delete();
   fDeActive->Delete();
}

//______________________________________________________________________________
void TMonitor::Activate(TSocket *sock)
{
   // Activate a de-activated socket.

   TIter next(fDeActive);
   TSocketHandler *s;

   while ((s = (TSocketHandler *) next())) {
      if (sock == s->GetSocket()) {
         fDeActive->Remove(s);
         fActive->Add(s);
         gSystem->AddFileHandler(s);
         return;
      }
   }
}

//______________________________________________________________________________
void TMonitor::ActivateAll()
{
   // Activate all de-activated sockets.

   TIter next(fDeActive);
   TSocketHandler *s;

   while ((s = (TSocketHandler *) next())) {
      fActive->Add(s);
      gSystem->AddFileHandler(s);
   }
   fDeActive->Clear();
}

//______________________________________________________________________________
void TMonitor::DeActivate(TSocket *sock)
{
   // De-activate a socket.

   TIter next(fActive);
   TSocketHandler *s;

   while ((s = (TSocketHandler *) next())) {
      if (sock == s->GetSocket()) {
         fActive->Remove(s);
         fDeActive->Add(s);
         s->Remove();
         return;
      }
   }
}

//______________________________________________________________________________
void TMonitor::DeActivateAll()
{
   // De-activate all activated sockets.

   TIter next(fActive);
   TSocketHandler *s;

   while ((s = (TSocketHandler *) next())) {
      fDeActive->Add(s);
      s->Remove();
   }
   fActive->Clear();
}

//______________________________________________________________________________
TSocket *TMonitor::Select()
{
   // Return pointer to socket for which an event is waiting.

   fReady = 0;

   while (!fReady)
      gSystem->InnerLoop();

   return fReady;
}

//______________________________________________________________________________
TSocket *TMonitor::Select(Long_t timeout)
{
   // Return pointer to socket for which an event is waiting. Wait a maximum
   // of timeout milliseconds. If return is due to timeout it returns
   // (TSocket *)-1.

   fReady = 0;

   TTimeOutTimer t(this, timeout);

   while (!fReady)
      gSystem->InnerLoop();

   return fReady;
}

//______________________________________________________________________________
void TMonitor::SetReady(TSocket *sock)
{
   // Called by TSocketHandler::Notify() to signal which socket is ready
   // to be read. User should not call this routine. The ready socket will
   // be returned via the Select() user function.

   fReady = sock;
}

//______________________________________________________________________________
Int_t TMonitor::GetActive() const
{
   // Return number of sockets in the active list.

   return fActive->GetSize();
}

//______________________________________________________________________________
Int_t TMonitor::GetDeActive() const
{
   // Return number of sockets in the de-active list.

   return fDeActive->GetSize();
}
