/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMPClient
#define ROOT_TMPClient

#include "TMonitor.h"
#include "TSocket.h"
#include "TClass.h"
#include "TCollection.h"
#include "TMPServer.h"
#include "TSysEvtHandler.h"
#include "MPSendRecv.h"
#include <vector>
#include <unistd.h> //pid_t
#include <iostream>
#include <memory> //unique_ptr

//////////////////////////////////////////////////////////////////////////
///
/// This is an implementation of a TSignalHandler that is added to the
/// eventloop in the children processes spawned by a TClient. It reacts
/// to SIGINT messages shutting down the worker and performing clean-up
/// operation before exiting. 
///
//////////////////////////////////////////////////////////////////////////

class TMPInterruptHandler : public TSignalHandler {
   ClassDef(TMPInterruptHandler, 0);
public:
   TMPInterruptHandler();
   Bool_t Notify();
};

//////////////////////////////////////////////////////////////////////////
///
/// Base class for multiprocess applications' clients. It provides a
/// simple interface to fork a ROOT session into "server" worker sessions
/// and exchange messages with them. Multiprocessing applications can build
/// on TMPClient and TMPServer.
///
//////////////////////////////////////////////////////////////////////////

class TMPClient {
public:
   explicit TMPClient(unsigned nWorkers = 0);
   ~TMPClient();
   //it doesn't make sense to copy a TMPClient
   TMPClient(const TMPClient&) = delete;
   TMPClient& operator=(const TMPClient&) = delete;

   bool Fork(TMPServer *server); //using a unique_ptr here would be cumbersome: passing unique_ptr is verbose and children must not delete server when leaving Fork's scope
   unsigned Broadcast(unsigned code, unsigned nMessages = 0);
   template<class T> unsigned Broadcast(unsigned code, const std::vector<T> &objs);
   template<class T> unsigned Broadcast(unsigned code, std::initializer_list<T> &objs);
   template<class T> unsigned Broadcast(unsigned code, T obj, unsigned nMessages = 0);
   inline TMonitor &GetMonitor() { return fMon; }
   inline bool GetIsParent() const { return fIsParent; }
   inline void SetNWorkers(unsigned n) { fNWorkers = n; }
   inline unsigned GetNWorkers() const { return fNWorkers; }
   void DeActivate(TSocket *s);
   void Remove(TSocket *s);
   void ReapServers();
   void HandleMPCode(MPCodeBufPair& msg, TSocket *sender);

private:
   bool fIsParent; ///< This is true if this is the parent/client process, false if this is a child/worker process
   static constexpr unsigned fPortN = 9090; ///< Number of the port over which the communication between client and workers happen
   std::vector<pid_t> fServerPids; ///< A vector containing the PIDs of children processes/workers
   TMonitor fMon; ///< A TMonitor is used to manage the sockets and detect socket events
   unsigned fNWorkers; ///< The number of workers that should be spawned upon forking (i.e. the number of times Fork will fork)
};


//////////////////////////////////////////////////////////////////////////
/// Send a message with a different object to each server.
/// The number of messages successfully sent is returned.
template<class T>
unsigned TMPClient::Broadcast(unsigned code, const std::vector<T> &args)
{
   fMon.ActivateAll();

   std::unique_ptr<TList> l(fMon.GetListOfActives());
   TIter nextSocket(l.get());
   TSocket *s = nullptr;
   unsigned count = 0;
   unsigned size = args.size();
   while ((s = (TSocket *)nextSocket()) && count < size) {
      if(MPSend(s, code, args[count])) {
         fMon.DeActivate(s);
         ++count;
      } else {
         std::cerr << "[E] Could not send message to server\n";
      }
   }

   return count;
}


//////////////////////////////////////////////////////////////////////////
/// Send a message with a different object to each server.
/// The number of messages successfully sent is returned.
template<class T>
unsigned TMPClient::Broadcast(unsigned code, std::initializer_list<T>& args)
{
   std::vector<T> vargs(std::move(args));
   return Broadcast(code, vargs);
}


//////////////////////////////////////////////////////////////////////////
/// Send a message containing obj to each worker, up to a maximum
/// of nMessages workers.
/// If nMessages = 0, send a message to every worker.
/// The number of messages successfully sent is returned.
template<class T>
unsigned TMPClient::Broadcast(unsigned code, T obj, unsigned nMessages) 
{
   if(!nMessages)
      nMessages = fNWorkers;
   unsigned count = 0;
   fMon.ActivateAll();

   //send message to all sockets
   std::unique_ptr<TList> l(fMon.GetListOfActives());
   TIter next(l.get());
   TSocket *s = nullptr;
   while ((s = (TSocket *)next()) && count < nMessages) {
      if(MPSend(s, code, obj)) {
         fMon.DeActivate(s);
         ++count;
      } else {
         std::cerr << "[E] Could not send message to server\n";
      }
   }

   return count;
}

#endif
