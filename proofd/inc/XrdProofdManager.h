// @(#)root/proofd:$Name:  $:$Id:$
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

#include "XrdProofdAux.h"
#include "XrdOuc/XrdOucPthread.hh"
#include "XrdOuc/XrdOucString.hh"

class XrdClientMessage;
class XrdProofWorker;
class XrdProofdResponse;

class XrdProofdManager {

 public:
   XrdProofdManager();
   virtual ~XrdProofdManager();

   XrdOucRecMutex   *Mutex() { return &fMutex; }

   int               Config(const char *fn, XrdOucError *e = 0);

   // List of available workers (on master only)
   std::list<XrdProofWorker *> *GetActiveWorkers();
   // Type of resource from which the info is taken
   int               ResourceType() const { return fResourceType; }

   // Node properties
   int               SrvType() const { return fSrvType; }
   const char       *EffectiveUser() const { return fEffectiveUser.c_str(); }
   const char       *Host() const { return fHost.c_str(); }
   int               Port() const { return fPort; }
   const char       *Image() const { return fImage.c_str(); }
   const char       *WorkDir() const { return fWorkDir.c_str(); }
   const char       *DataSetDir() const { return fDataSetDir.c_str(); }

   // This part may evolve in the future due to better understanding of
   // how resource brokering will work; for the time being we just move in
   // here the functionality we have now
   int               Broadcast(int type, const char *msg, XrdProofdResponse *r);
   XrdClientMessage *Send(const char *url, int type,
                          const char *msg, int srvtype, XrdProofdResponse *r);

   const char       *PROOFcfg() const { return fPROOFcfg.fName.c_str(); }

 private:
   XrdOucRecMutex    fMutex;        // Atomize this instance

   XrdProofdFile     fCfgFile;      // Configuration file

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

   std::list<XrdProofWorker *> fWorkers;  // vector of possible workers

   XrdOucError      *fEDest;        // Error message handler

   void              CreateDefaultPROOFcfg();
   int               ReadPROOFcfg();
};

#endif
