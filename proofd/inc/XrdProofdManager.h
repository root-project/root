// @(#)root/proofd:$Id$
// Author: G. Ganis June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdManager
#define ROOT_XrdProofdManager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdManager                                                     //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Class mapping manager fonctionality.                                 //
// On masters it keeps info about the available worker nodes and allows //
// communication with them. In particular, it reads the proof.conf file //
// when working with static resources.                                  //
// On workers it handles the communication with the master              //
// (to be implemented).                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <list>

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucPthread.hh"
#else
#  include "XrdSys/XrdSysPthread.hh"
#endif
#include "XrdProofdAux.h"
#include "XrdProofConn.h"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"

class XrdClientMessage;
class XrdProofWorker;
class XrdProofdResponse;
class XrdProofServProxy;

class XrdProofdManager {

 public:
   XrdProofdManager();
   virtual ~XrdProofdManager();

   XrdSysRecMutex   *Mutex() { return &fMutex; }

   int               Config(const char *fn, XrdSysError *e = 0);

   // List of available workers (on master only)
   std::list<XrdProofWorker *> *GetActiveWorkers();
   // List of unique nodes (on master only)
   std::list<XrdProofWorker *> *GetNodes();
   // Type of resource from which the info is taken
   int               ResourceType() const { return fResourceType; }

   // Keping track of active sessions
   std::list<XrdProofServProxy *> *GetActiveSessions() { XrdSysMutexHelper mhp(&fMutex);
                                                         return &fActiveSessions; }
   void              AddActiveSession(XrdProofServProxy *p) { XrdSysMutexHelper mhp(&fMutex);
                                                              fActiveSessions.push_back(p); }
   XrdProofServProxy *GetActiveSession(int pid);
   void              RemoveActiveSession(XrdProofServProxy *p) { XrdSysMutexHelper mhp(&fMutex);
                                                                 fActiveSessions.remove(p); }
   // Connections to other xrootd running XrdProofdProtocols
   XrdProofConn     *GetProofConn(const char *url);

   // Node properties
   int               SrvType() const { return fSrvType; }
   const char       *EffectiveUser() const { return fEffectiveUser.c_str(); }
   const char       *Host() const { return fHost.c_str(); }
   int               Port() const { return fPort; }
   const char       *Image() const { return fImage.c_str(); }
   const char       *WorkDir() const { return fWorkDir.c_str(); }
   const char       *DataSetDir() const { return fDataSetDir.c_str(); }

   bool              IsSuperMst() const { return fSuperMst; }

   // This part may evolve in the future due to better understanding of
   // how resource brokering will work; for the time being we just move in
   // here the functionality we have now
   int               Broadcast(int type, const char *msg, XrdProofdResponse *r);
   XrdClientMessage *Send(const char *url, int type,
                          const char *msg, int srvtype, XrdProofdResponse *r);

   const char       *PROOFcfg() const { return fPROOFcfg.fName.c_str(); }

 private:
   XrdSysRecMutex    fMutex;        // Atomize this instance

   XrdProofdFile     fCfgFile;      // Configuration file
   bool              fSuperMst;     // true if this node is a SuperMst

   int               fSrvType;      // Master, Submaster, Worker or any
   XrdOucString      fEffectiveUser;  // Effective user
   XrdOucString      fHost;         // local host name
   int               fPort;         // Port for client-like connections
   XrdOucString      fImage;        // image name for these servers
   XrdOucString      fWorkDir;      // working dir for these servers
   XrdOucString      fDataSetDir;   // dataset dir for this master server
   int               fNumLocalWrks; // Number of workers to be started locally
   int               fResourceType; // resource type
   XrdProofdFile     fPROOFcfg;     // PROOF static configuration

   std::list<XrdProofWorker *> fWorkers;  // List of possible workers
   std::list<XrdProofWorker *> fNodes;   // List of worker unique nodes

   std::list<XrdProofServProxy *> fActiveSessions; // List of active sessions (non-idle)

   XrdSysError      *fEDest;        // Error message handler

   XrdOucHash<XrdProofConn> fProofConnHash; // Available connections

   void              CreateDefaultPROOFcfg();
   int               ReadPROOFcfg();
};

#endif
