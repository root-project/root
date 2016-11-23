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

#include "MPSendRecv.h"
#include "TMonitor.h"
#include "TMPWorker.h"
#include <memory> //unique_ptr
#include <iostream>
#include <unistd.h> //pid_t
#include <vector>

class TMPClient {
public:
   explicit TMPClient(unsigned nWorkers = 0);
   ~TMPClient();
   //it doesn't make sense to copy a TMPClient
   TMPClient(const TMPClient &) = delete;
   TMPClient &operator=(const TMPClient &) = delete;

   bool Fork(TMPWorker &server); // we expect application to pass a reference to an inheriting class and take advantage of polymorphism
   unsigned Broadcast(unsigned code, unsigned nMessages = 0);
   template<class T> unsigned Broadcast(unsigned code, const std::vector<T> &objs);
   template<class T> unsigned Broadcast(unsigned code, std::initializer_list<T> &objs);
   template<class T> unsigned Broadcast(unsigned code, T obj, unsigned nMessages = 0);
   TMonitor &GetMonitor() { return fMon; }
   bool GetIsParent() const { return fIsParent; }
   /// Set the number of workers that will be spawned by the next call to Fork()
   void SetNWorkers(unsigned n) { fNWorkers = n; }
   unsigned GetNWorkers() const { return fNWorkers; }
   void DeActivate(TSocket *s);
   void Remove(TSocket *s);
   void ReapWorkers();
   void HandleMPCode(MPCodeBufPair &msg, TSocket *sender);

private:
   bool fIsParent; ///< This is true if this is the parent/client process, false if this is a child/worker process
   std::vector<pid_t> fWorkerPids; ///< A vector containing the PIDs of children processes/workers
   TMonitor fMon; ///< This object manages the sockets and detect socket events via TMonitor::Select
   unsigned fNWorkers; ///< The number of workers that should be spawned upon forking
};


//////////////////////////////////////////////////////////////////////////
/// Send a message with a different object to each server.
/// Sockets can either be in an "active" or "non-active" state. This method
/// activates all the sockets through which the client is connected to the
/// workers, and deactivates them when a message is sent to the corresponding
/// worker. This way the sockets pertaining to workers who have been left
/// idle will be the only ones in the active list
/// (TSocket::GetMonitor()->GetListOfActives()) after execution.
/// \param code the code of the message to send (e.g. EMPCode)
/// \param args
/// \parblock
/// a vector containing the different messages to be sent. If the size of
/// the vector is smaller than the number of workers, a message will be
/// sent only to the first args.size() workers. If the size of the args vector
/// is bigger than the number of workers, only the first fNWorkers arguments
/// will be sent.
/// \endparblock
/// \return the number of messages successfully sent
template<class T>
unsigned TMPClient::Broadcast(unsigned code, const std::vector<T> &args)
{
   fMon.ActivateAll();

   std::unique_ptr<TList> lp(fMon.GetListOfActives());
   unsigned count = 0;
   unsigned nArgs = args.size();
   for (auto s : *lp) {
      if (count == nArgs)
         break;
      if (MPSend((TSocket *)s, code, args[count])) {
         fMon.DeActivate((TSocket *)s);
         ++count;
      } else {
         Error("TMPClient::Broadcast", "[E] Could not send message to server\n");
      }
   }

   return count;
}


//////////////////////////////////////////////////////////////////////////
/// Send a message with a different object to each server.
/// See TMPClient::Broadcast(unsigned code, const std::vector<T> &args)
/// for more informations.
template<class T>
unsigned TMPClient::Broadcast(unsigned code, std::initializer_list<T> &args)
{
   std::vector<T> vargs(std::move(args));
   return Broadcast(code, vargs);
}


//////////////////////////////////////////////////////////////////////////
/// Send a message containing code and obj to each worker, up to a
/// maximum number of nMessages workers. See
/// Broadcast(unsigned code, unsigned nMessages) for more informations.
/// \param code the code of the message to send (e.g. EMPCode)
/// \param obj the object to send
/// \param nMessages
/// \parblock
/// the maximum number of messages to send.
/// If nMessages == 0, send a message to every worker.
/// \endparblock
/// \return the number of messages successfully sent
template<class T>
unsigned TMPClient::Broadcast(unsigned code, T obj, unsigned nMessages)
{
   if (nMessages == 0)
      nMessages = fNWorkers;
   unsigned count = 0;
   fMon.ActivateAll();

   //send message to all sockets
   std::unique_ptr<TList> lp(fMon.GetListOfActives());
   for (auto s : *lp) {
      if (count == nMessages)
         break;
      if (MPSend((TSocket *)s, code, obj)) {
         fMon.DeActivate((TSocket *)s);
         ++count;
      } else {
         Error("TMPClient::Broadcast", "[E] Could not send message to server\n");
      }
   }

   return count;
}

#endif
