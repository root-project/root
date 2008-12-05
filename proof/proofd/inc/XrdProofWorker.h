// @(#)root/proofd:$Id$
// Author: Gerardo Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofWorker
#define ROOT_XrdProofWorker

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofWorker                                                       //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Class with information about a potential worker.                     //
// A list of instances of this class is built using the config file or  //
// or the information collected from the resource discoverers.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <list>

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucPthread.hh"
#else
#  include "XrdSys/XrdSysPthread.hh"
#endif
#include "XrdOuc/XrdOucString.hh"


class XrdProofdProofServ;

class XrdProofWorker {

 public:
   XrdProofWorker(const char *str = 0);
   virtual ~XrdProofWorker();

   void                    Reset(const char *str); // Set from 'str'

   const char             *Export();

   bool                    Matches(const char *host);
   bool                    Matches(XrdProofWorker *wrk);
   int                     GetNActiveSessions();

   static void             Sort(std::list<XrdProofWorker *> *lst,
                                bool (*f)(XrdProofWorker *&lhs,
                                          XrdProofWorker *&rhs));
   inline int              Active() const { XrdSysMutexHelper mhp(fMutex); return fActive; }
   inline void             CountActive(int n) { XrdSysMutexHelper mhp(fMutex); fActive += n; }
   inline int              Suspended() const { XrdSysMutexHelper mhp(fMutex); return fSuspended; }
   inline void             CountSuspended(int n) { XrdSysMutexHelper mhp(fMutex); fSuspended += n; }

   std::list<XrdProofdProofServ *> fProofServs; // ProofServ sessions using
                                               // this worker

   // Worker definitions
   XrdOucString            fExport;      // export string
   char                    fType;        // type: worker ('W') or submaster ('S')
   XrdOucString            fUser;        // user
   XrdOucString            fHost;        // host FQDN
   int                     fPort;        // port
   int                     fPerfIdx;     // performance index
   XrdOucString            fImage;       // image name
   XrdOucString            fWorkDir;     // work directory
   XrdOucString            fMsd;         // mass storage domain
   XrdOucString            fId;          // ID string

private:
   XrdSysRecMutex         *fMutex;       // Local mutex
   // Counters
   int                     fActive;      // number of active sessions
   int                     fSuspended;   // number of suspended sessions 
};

#endif
