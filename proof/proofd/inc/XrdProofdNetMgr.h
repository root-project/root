// @(#)root/proofd:$Id$
// Author: G. Ganis  Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdNetMgr
#define ROOT_XrdProofdNetMgr

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdNetMgr                                                     //
//                                                                      //
// Authors: G. Ganis, CERN, 2008                                        //
//                                                                      //
// Manages connections between PROOF server daemons                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucPthread.hh"
#else
#  include "XrdSys/XrdSysPthread.hh"
#endif

#if defined(BUILD_BONJOUR)
#include "XrdOuc/XrdOucBonjour.hh"
#endif
#include "XrdOuc/XrdOucHash.hh"

#include "XrdProofConn.h"
#include "XrdProofdConfig.h"

class XrdProofdDirective;
class XrdProofdManager;
class XrdProofdProtocol;
class XrdProofdResponse;
class XrdProofWorker;

class XrdProofdNetMgr : public XrdProofdConfig {

private:

   XrdSysRecMutex     fMutex;          // Atomize this instance

   XrdProofdManager  *fMgr;
   XrdOucHash<XrdProofConn> fProofConnHash;            // Available connections
   int                fNumLocalWrks;   // Number of workers to be started locally
   int                fResourceType;   // resource type
   XrdProofdFile      fPROOFcfg;       // PROOF static configuration
   bool               fReloadPROOFcfg; // Whether the file should regurarl checked for updates
   bool               fDfltFallback;   // Whether to fallback to default if file cannot be read
   bool               fWorkerUsrCfg;   // user cfg files enabled / disabled
   int                fRequestTO;      // Timeout on broadcast request

   std::list<XrdProofWorker *> fDfltWorkers; // List of possible default workers
   std::list<XrdProofWorker *> fRegWorkers;  // List of all workers registered
   std::list<XrdProofWorker *> fWorkers;     // List of currently available workers
   std::list<XrdProofWorker *> fNodes;       // List of worker unique nodes

   void               CreateDefaultPROOFcfg();
   int                ReadPROOFcfg(bool reset = 1);
   int                FindUniqueNodes();

   int                LocateLocalFile(XrdOucString &file);

   int                DoDirectiveBonjour(char *val, XrdOucStream *cfg, bool);
   int                DoDirectiveAdminReqTO(char *, XrdOucStream *, bool);
   int                DoDirectiveResource(char *, XrdOucStream *, bool);
   int                DoDirectiveWorker(char *, XrdOucStream *, bool);

   bool               fBonjourEnabled;
#if defined(BUILD_BONJOUR)
   int                fBonjourRequestedSrvType; // Register, Discover or Both.
   XrdOucBonjour     *fBonjourManager; // A reference to the Bonjour manager.
   XrdOucString       fBonjourServiceType;
   XrdOucString       fBonjourName;
   XrdOucString       fBonjourDomain;
   int                fBonjourCores;
   int                LoadBonjourModule(int srvtype);
   static void *      ProcessBonjourUpdate(void * context);
#endif

public:
   XrdProofdNetMgr(XrdProofdManager *mgr, XrdProtocol_Config *pi, XrdSysError *e);
   virtual ~XrdProofdNetMgr();

   int                Config(bool rcf = 0);
   int                DoDirective(XrdProofdDirective *d,
                                  char *val, XrdOucStream *cfg, bool rcf);
   void               RegisterDirectives();

   void               Dump();

   const char        *PROOFcfg() const { return fPROOFcfg.fName.c_str(); }
   bool               WorkerUsrCfg() const { return fWorkerUsrCfg; }

   int                Broadcast(int type, const char *msg, const char *usr = 0,
                                XrdProofdResponse *r = 0, bool notify = 0, int subtype = -1);
   int                BroadcastCtrlC(const char *usr);
   XrdProofConn      *GetProofConn(const char *url);
   bool               IsLocal(const char *host, bool checkport = 0);
   XrdClientMessage  *Send(const char *url, int type,
                           const char *msg, int srvtype, XrdProofdResponse *r,
                           bool notify = 0, int subtype = -1);

   int                ReadBuffer(XrdProofdProtocol *p);
   char              *ReadBufferLocal(const char *file, kXR_int64 ofs, int &len);
   char              *ReadBufferLocal(const char *file, const char *pat, int &len, int opt);
   char              *ReadBufferRemote(const char *url, const char *file,
                                       kXR_int64 ofs, int &len, int grep);
   char              *ReadLogPaths(const char *url, const char *stag, int isess);
   char              *ReadLogPaths(const char *stag, int isess);

   // List of available and unique workers (on master only)
   std::list<XrdProofWorker *> *GetActiveWorkers();
   std::list<XrdProofWorker *> *GetNodes();

#if defined(BUILD_BONJOUR)
   // Interface of Bonjour services.
   int                GetBonjourRequestedServiceType() const { return fBonjourRequestedSrvType; }
   const char        *GetBonjourServiceType() const { return (fBonjourServiceType.length()) ? fBonjourServiceType.c_str() : "_proof._tcp."; }
   const char        *GetBonjourName() const { return (fBonjourName.length()) ? fBonjourName.c_str() : NULL; }
   const char        *GetBonjourDomain() const { return (fBonjourDomain.length()) ? fBonjourDomain.c_str() : NULL; }
   int                GetBonjourCores() const { return fBonjourCores; }
   static bool        CheckBonjourRoleCoherence(int role, int bonjourSrvType);
#endif
   void               BalanceNodesOrder();
};

// Auxiliary structure to store information for the balancer algorithm.
typedef struct BalancerInfo {
   unsigned int available;
   unsigned int per_iteration;
   unsigned int added;
} BalancerInfo;

#endif
