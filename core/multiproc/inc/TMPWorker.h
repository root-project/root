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

#include "MPCode.h"
#include "MPSendRecv.h" //MPCodeBufPair
#include "PoolUtils.h"
#include "TSysEvtHandler.h" //TFileHandler

#include <memory> //unique_ptr
#include <string>
#include <sstream>
#include <type_traits> //std::result_of
#include <unistd.h> //pid_t

class TMPWorker {
public:
   TMPWorker() : fNWorkers(0), fMaxNEntries(0),
                 fProcessedEntries(0), fS(), fPid(0), fNWorker(0) { }
   TMPWorker(unsigned nWorkers, ULong64_t maxEntries)
               : fNWorkers(nWorkers), fMaxNEntries(maxEntries),
                 fProcessedEntries(0), fS(), fPid(0), fNWorker(0) { }
   virtual ~TMPWorker() { }
   //it doesn't make sense to copy a TMPWorker (each one has a uniq_ptr to its socket)
   TMPWorker(const TMPWorker &) = delete;
   TMPWorker &operator=(const TMPWorker &) = delete;

   virtual void Init(int fd, unsigned workerN);
   void Run();
   TSocket *GetSocket() { return fS.get(); }
   pid_t GetPid() { return fPid; }
   unsigned GetNWorker() const { return fNWorker; }

protected:
   std::string fId; ///< identifier string in the form W<nwrk>|P<proc id>
   unsigned fNWorkers; ///< the number of workers spawned
   ULong64_t fMaxNEntries; ///< the maximum number of entries to be processed by this worker
   ULong64_t fProcessedEntries; ///< the number of entries processed by this worker so far

   void   SendError(const std::string& errmsg, unsigned int code = MPCode::kError);

private:
   virtual void HandleInput(MPCodeBufPair &msg);

   std::unique_ptr<TSocket> fS; ///< This worker's socket. The unique_ptr makes sure resources are released.
   pid_t fPid; ///< the PID of the process in which this worker is running
   unsigned fNWorker; ///< the ordinal number of this worker (0 to nWorkers-1)
};

#endif
