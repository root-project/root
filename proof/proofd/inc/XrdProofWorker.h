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

   const char             *Export(const char *ord = 0);

   bool                    Matches(const char *host);
   bool                    Matches(XrdProofWorker *wrk);
   int                     GetNActiveSessions();

   static void             Sort(std::list<XrdProofWorker *> *lst,
                                bool (*f)(XrdProofWorker *&lhs,
                                          XrdProofWorker *&rhs));

   inline int              Active() const {
      XrdSysMutexHelper mhp(fMutex);
      return fProofServs.size();
   }
   inline void             AddProofServ(XrdProofdProofServ *xps) {
      XrdSysMutexHelper mhp(fMutex);
      return fProofServs.push_back(xps);
   }
   inline void             RemoveProofServ(XrdProofdProofServ *xps) {
      XrdSysMutexHelper mhp(fMutex);
      return fProofServs.remove(xps);
   }
   // Allows to copy the session objects from other worker.
   void                    MergeProofServs(const XrdProofWorker &other);

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

   bool                    fActive;      // TRUE if available

private:
   XrdSysRecMutex         *fMutex;       // Local mutex
};

#endif
