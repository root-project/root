/* @(#)root/multiproc:$Id$ */
// Author: Enrico Guiraud July 2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMPWorker
#define ROOT_TMPWorker

#include "TSysEvtHandler.h" //TFileHandler
#include "MPSendRecv.h" //MPCodeBufPair
#include <unistd.h> //pid_t
#include <memory> //unique_ptr

class TMPWorker {
   /// \cond
   ClassDef(TMPWorker, 0);
   /// \endcond
public:
   TMPWorker();
   virtual ~TMPWorker() {};
   //it doesn't make sense to copy a TMPWorker (each one has a uniq_ptr to its socket)
   TMPWorker(const TMPWorker &) = delete;
   TMPWorker &operator=(const TMPWorker &) = delete;

   virtual void Init(int fd, unsigned workerN);
   void Run();
   TSocket *GetSocket() { return fS.get(); }
   pid_t GetPid() { return fPid; }
   unsigned GetNWorker() const { return fNWorker; }


private:
   virtual void HandleInput(MPCodeBufPair &msg);

   std::unique_ptr<TSocket> fS; ///< This worker's socket. The unique_ptr makes sure resources are released.
   pid_t fPid; ///< the PID of the process in which this worker is running
   unsigned fNWorker; ///< the ordinal number of this worker (0 to nWorkers-1)
};

#endif
