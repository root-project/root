// @(#)root/net:$Id$
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
// This utility class is only used by TMonitor.
//

class TSocketHandler : public TFileHandler {
private:
   TMonitor  *fMonitor;   //monitor to which this handler belongs
   TSocket   *fSocket;    //socket being handled

public:
   TSocketHandler(TMonitor *m, TSocket *s, Int_t interest, Bool_t mainloop = kTRUE);
   Bool_t   Notify();
   Bool_t   ReadNotify() { return Notify(); }
   Bool_t   WriteNotify() { return Notify(); }
   TSocket *GetSocket() const { return fSocket; }
};

TSocketHandler::TSocketHandler(TMonitor *m, TSocket *s,
                               Int_t interest, Bool_t mainloop)
               : TFileHandler(s->GetDescriptor(), interest)
{
   //constructor
   fMonitor = m;
   fSocket  = s;

   if (mainloop)
      Add();
}

Bool_t TSocketHandler::Notify()
{
   //notifier
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
   //constructor
   fMonitor = m;
   gSystem->AddTimer(this);
}

Bool_t TTimeOutTimer::Notify()
{
   //notifier
   fMonitor->SetReady((TSocket *)-1);
   Remove();       // one shot only
   return kTRUE;
}
//------------------------------------------------------------------------------


ClassImp(TMonitor);

////////////////////////////////////////////////////////////////////////////////
/// Create a monitor object. If mainloop is true the monitoring will be
/// done in the main event loop.

TMonitor::TMonitor(Bool_t mainloop) : TObject() , TQObject()
{
   R__ASSERT(gSystem);

   fActive    = new TList;
   fDeActive  = new TList;
   fMainLoop  = mainloop;
   fInterrupt = kFALSE;
   fReady     = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

TMonitor::TMonitor(const TMonitor &m) : TObject() , TQObject()
{
   TSocketHandler *sh = 0;
   // Active list
   fActive   = new TList;
   TIter nxa(m.fActive);
   while ((sh = (TSocketHandler *)nxa())) {
      Int_t mask = 0;
      if (sh->HasReadInterest()) mask |= 0x1;
      if (sh->HasWriteInterest()) mask |= 0x2;
      fActive->Add(new TSocketHandler(this, sh->GetSocket(), mask, m.fMainLoop));
   }
   // Deactive list
   fDeActive = new TList;
   TIter nxd(m.fDeActive);
   while ((sh = (TSocketHandler *)nxd())) {
      Int_t mask = 0;
      if (sh->HasReadInterest()) mask |= 0x1;
      if (sh->HasWriteInterest()) mask |= 0x2;
      fDeActive->Add(new TSocketHandler(this, sh->GetSocket(), mask, m.fMainLoop));
   }
   // Other members
   fMainLoop = m.fMainLoop;
   fInterrupt = m.fInterrupt;
   fReady = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup the monitor object. Does not delete sockets being monitored.

TMonitor::~TMonitor()
{
   fActive->Delete();
   SafeDelete(fActive);

   fDeActive->Delete();
   SafeDelete(fDeActive);
}

////////////////////////////////////////////////////////////////////////////////
/// Add socket to the monitor's active list. If interest=kRead then we
/// want to monitor the socket for read readiness, if interest=kWrite
/// then we monitor the socket for write readiness, if interest=kRead|kWrite
/// then we monitor both read and write readiness.

void TMonitor::Add(TSocket *sock, Int_t interest)
{
   fActive->Add(new TSocketHandler(this, sock, interest, fMainLoop));
}

////////////////////////////////////////////////////////////////////////////////
/// Set interest mask for socket sock to interest. If the socket is not
/// in the active list move it or add it there.
/// If interest=kRead then we want to monitor the socket for read readiness,
/// if interest=kWrite then we monitor the socket for write readiness,
/// if interest=kRead|kWrite then we monitor both read and write readiness.

void TMonitor::SetInterest(TSocket *sock, Int_t interest)
{
   TSocketHandler *s = 0;

   if (!interest)
      interest = kRead;

   // Check first the activated list ...
   TIter next(fActive);
   while ((s = (TSocketHandler *) next())) {
      if (sock == s->GetSocket()) {
         s->SetInterest(interest);
         return;
      }
   }

   // Check now the deactivated list ...
   TIter next1(fDeActive);
   while ((s = (TSocketHandler *) next1())) {
      if (sock == s->GetSocket()) {
         fDeActive->Remove(s);
         fActive->Add(s);
         s->SetInterest(interest);
         return;
      }
   }

   // The socket is not in our lists: just add it
   fActive->Add(new TSocketHandler(this, sock, interest, fMainLoop));
}

////////////////////////////////////////////////////////////////////////////////
/// Remove a socket from the monitor.

void TMonitor::Remove(TSocket *sock)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Remove all sockets from the monitor.

void TMonitor::RemoveAll()
{
   fActive->Delete();
   fDeActive->Delete();
}

////////////////////////////////////////////////////////////////////////////////
/// Activate a de-activated socket.

void TMonitor::Activate(TSocket *sock)
{
   TIter next(fDeActive);
   TSocketHandler *s;

   while ((s = (TSocketHandler *) next())) {
      if (sock == s->GetSocket()) {
         fDeActive->Remove(s);
         fActive->Add(s);
         s->Add();
         return;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Activate all de-activated sockets.

void TMonitor::ActivateAll()
{
   TIter next(fDeActive);
   TSocketHandler *s;

   while ((s = (TSocketHandler *) next())) {
      fActive->Add(s);
      s->Add();
   }
   fDeActive->Clear();
   fInterrupt = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// De-activate a socket.

void TMonitor::DeActivate(TSocket *sock)
{
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

////////////////////////////////////////////////////////////////////////////////
/// De-activate all activated sockets.

void TMonitor::DeActivateAll()
{
   TIter next(fActive);
   TSocketHandler *s;

   while ((s = (TSocketHandler *) next())) {
      fDeActive->Add(s);
      s->Remove();
   }
   fActive->Clear();
   fInterrupt = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Return pointer to socket for which an event is waiting.
/// Select can be interrupt by a call to Interrupt() (e.g. connected with a
/// Ctrl-C handler); a call to ResetInterrupt() before Select() is advisable
/// in such a case.
/// Return 0 in case of error.

TSocket *TMonitor::Select()
{
   fReady = 0;

   while (!fReady && !fInterrupt)
      gSystem->InnerLoop();

   // Notify interrupts
   if (fInterrupt) {
      fInterrupt = kFALSE;
      fReady = 0;
      Info("Select","*** interrupt occured ***");
   }

   return fReady;
}

////////////////////////////////////////////////////////////////////////////////
/// Return pointer to socket for which an event is waiting.
/// Wait a maximum of timeout milliseconds.
/// If return is due to timeout it returns (TSocket *)-1.
/// Select() can be interrupt by a call to Interrupt() (e.g. connected with a
/// Ctrl-C handler); a call to ResetInterrupt() before Select() is advisable
/// in such a case.
/// Return 0 in case of any other error situation.

TSocket *TMonitor::Select(Long_t timeout)
{
   if (timeout < 0)
      return TMonitor::Select();

   fReady = 0;

   TTimeOutTimer t(this, timeout);

   while (!fReady && !fInterrupt)
      gSystem->InnerLoop();

   // Notify interrupts
   if (fInterrupt) {
      fInterrupt = kFALSE;
      fReady = 0;
      Info("Select","*** interrupt occured ***");
   }

   return fReady;
}

////////////////////////////////////////////////////////////////////////////////
/// Return numbers of sockets that are ready for reading or writing.
/// Wait a maximum of timeout milliseconds.
/// Return 0 if timed-out. Return < 0 in case of error.
/// If rdready and/or wrready are not 0, the lists of sockets with
/// something to read and/or write are also returned.

Int_t TMonitor::Select(TList *rdready, TList *wrready, Long_t timeout)
{
   Int_t nr = -2;

   TSocketHandler *h = 0;
   Int_t ns = fActive->GetSize();
   if (ns == 1) {
      // Avoid additional loops inside
      h = (TSocketHandler *)fActive->First();
      nr = gSystem->Select((TFileHandler *)h, timeout);
   } else if (ns > 1) {
      nr = gSystem->Select(fActive, timeout);
   }

   if (nr > 0 && (rdready || wrready)) {
      // Clear the lists
      if (rdready)
         rdready->Clear();
      if (wrready)
         wrready->Clear();
      // Got a file descriptor
      if (!h) {
         TIter next(fActive);
         while ((h = (TSocketHandler *)next())) {
            if (rdready && h->IsReadReady())
               rdready->Add(h->GetSocket());
            if (wrready && h->IsWriteReady())
               wrready->Add(h->GetSocket());
         }
      } else {
         if (rdready && h->IsReadReady())
            rdready->Add(h->GetSocket());
         if (wrready && h->IsWriteReady())
            wrready->Add(h->GetSocket());
      }
   }

   return nr;
}

////////////////////////////////////////////////////////////////////////////////
/// Called by TSocketHandler::Notify() to signal which socket is ready
/// to be read or written. User should not call this routine. The ready
/// socket will be returned via the Select() user function.
/// The Ready(TSocket *sock) signal is emitted.

void TMonitor::SetReady(TSocket *sock)
{
   fReady = sock;
   Ready(fReady);
}

////////////////////////////////////////////////////////////////////////////////
/// Return number of sockets in the active list. If timeout > 0, remove from
/// the list those sockets which did not have any activity since timeout
/// millisecs. If timeout = 0, then reset activity timestamp on all active
/// sockets. This time out is typically used if GetActive() is used to see
/// how many remotes still need to send something. If they pass the timeout
/// they will be skipped and GetActive() will return 0 and the loop can be
/// exited.

Int_t TMonitor::GetActive(Long_t timeout) const
{
   if (timeout >= 0) {
      TIter next(fActive);
      TSocketHandler *s;
      if (timeout > 0) {
         TTimeStamp now;
         while ((s = (TSocketHandler *) next())) {
            TSocket *xs = s->GetSocket();
            TTimeStamp ts = xs->GetLastUsage();
            Long_t dt = (Long_t)(now.GetSec() - ts.GetSec()) * 1000 +
                        (Long_t)(now.GetNanoSec() - ts.GetNanoSec()) / 1000000 ;
            if (dt > timeout) {
               Info("GetActive", "socket: %p: %s:%d did not show any activity"
                                 " during the last %ld millisecs: deactivating",
                                 xs, xs->GetInetAddress().GetHostName(),
                                 xs->GetInetAddress().GetPort(), timeout);
               fActive->Remove(s);
               fDeActive->Add(s);
               s->Remove();
            }
         }
      } else if (timeout == 0) {
         // Reset time stamps
         while ((s = (TSocketHandler *) next())) {
            s->GetSocket()->Touch();
         }
      }
   }
   return fActive->GetSize();
}

////////////////////////////////////////////////////////////////////////////////
/// Return number of sockets in the de-active list.

Int_t TMonitor::GetDeActive() const
{
   return fDeActive->GetSize();
}

////////////////////////////////////////////////////////////////////////////////
/// Check if socket 's' is in the active list. Avoids the duplication
/// of active list via TMonitor::GetListOfActives().

Bool_t TMonitor::IsActive(TSocket *sock) const
{
   TIter next(fActive);
   while (TSocketHandler *h = (TSocketHandler*) next())
      if (sock == h->GetSocket())
         return kTRUE;

   // Not found
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a list with all active sockets. This list must be deleted
/// by the user. DO NOT call Delete() on this list as it will delete
/// the sockets that are still being used by the monitor.

TList *TMonitor::GetListOfActives() const
{
   TList *list = new TList;

   TIter next(fActive);

   while (TSocketHandler *h = (TSocketHandler*) next())
      list->Add(h->GetSocket());

   return list;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns a list with all de-active sockets. This list must be deleted
/// by the user. DO NOT call Delete() on this list as it will delete
/// the sockets that are still being used by the monitor.

TList *TMonitor::GetListOfDeActives() const
{
   TList *list = new TList;

   TIter next(fDeActive);

   while (TSocketHandler *h = (TSocketHandler*) next())
      list->Add(h->GetSocket());

   return list;
}

////////////////////////////////////////////////////////////////////////////////
/// Emit signal when some socket is ready

void TMonitor::Ready(TSocket *sock)
{
   Emit("Ready(TSocket*)", (Longptr_t)sock);
}
