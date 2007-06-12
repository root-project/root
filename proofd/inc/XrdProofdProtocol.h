// @(#)root/proofd:$Name:  $:$Id: XrdProofdProtocol.h,v 1.23 2007/04/19 09:27:55 rdm Exp $
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdProtocol
#define ROOT_XrdProofdProtocol

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProtocol                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// XrdProtocol implementation to coordinate 'proofserv' applications.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdOuc/XrdOucError.hh"
#include "XrdOuc/XrdOucPthread.hh"
#include "XrdOuc/XrdOucSemWait.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSec/XrdSecInterface.hh"

#include "Xrd/XrdProtocol.hh"
#include "Xrd/XrdObject.hh"

#include "XProofProtocol.h"
#include "XrdProofdResponse.h"
#include "XrdProofServProxy.h"
#include "XrdProofdAux.h"

#include <list>
#include <vector>

// Version index: start from 1001 (0x3E9) to distinguish from 'proofd'
// To be increment when non-backward compatible changes are introduced
//  1001 (0x3E9) -> 1002 (0x3EA) : support for flexible env setting
#define XPROOFD_VERSBIN 0x000003EA
#define XPROOFD_VERSION "0.2"

#define XPD_LOGGEDIN       1
#define XPD_NEED_AUTH      2
#define XPD_ADMINUSER      4
#define XPD_NEED_MAP       8

enum EResourceType { kRTStatic, kRTPlb };
enum EStaticSelOpt { kSSORoundRobin, kSSORandom };

class XrdROOT;
class XrdBuffer;
class XrdClientMessage;
class XrdLink;
class XrdOucError;
class XrdOucTrace;
class XrdProofdClient;
class XrdProofdPInfo;
class XrdProofdPriority;
class XrdProofWorker;
class XrdScheduler;
class XrdSrvBuffer;

class XrdProofdProtocol : XrdProtocol {

friend class XrdProofdClient;
friend class XrdROOT;

public:
   XrdProofdProtocol();
   virtual ~XrdProofdProtocol() {} // Never gets destroyed

   static int    Configure(char *parms, XrdProtocol_Config *pi);
   void          DoIt() {}
   XrdProtocol  *Match(XrdLink *lp);
   int           Process(XrdLink *lp);
   void          Recycle(XrdLink *lp, int x, const char *y);
   int           Stats(char *buff, int blen, int do_sync);

   const char   *GetID() const { return (const char *)fClientID; }

   static int    Reconfig();
   static int    TrimTerminatedProcesses();

 private:

   int           Admin();
   int           Attach();
   int           Auth();
   int           Create();
   int           Destroy();
   int           Detach();
   int           GetBuff(int quantum);
   int           GetData(const char *dtype, char *buff, int blen);
   XrdProofServProxy *GetServer(int psid);
   int           Interrupt();
   int           Login();
   int           MapClient(bool all = 1);
   int           Ping();
   int           Process2();
   int           ReadBuffer();
   char         *ReadBufferLocal(const char *file, kXR_int64 ofs, int &len);
   char         *ReadBufferRemote(const char *file, kXR_int64 ofs, int &len);
   void          Reset();
   int           SendData(XrdProofdResponse *resp, kXR_int32 sid = -1, XrdSrvBuffer **buf = 0);
   int           SendDataN(XrdProofServProxy *xps, XrdSrvBuffer **buf = 0);
   int           SendMsg();
   int           SetUserEnvironment();
   int           Urgent();

   int           Broadcast(int type, const char *msg);
   bool          CanDoThis(const char *client);
   int           CleanupProofServ(bool all = 0, const char *usr = 0);
   int           KillProofServ(int pid, bool forcekill = 0, bool add = 1);
   int           KillProofServ(XrdProofServProxy *xps, bool forcekill = 0, bool add = 1);
   XrdClientMessage *SendCoordinator(const char *url, int type, const char *msg, int srvtype);
   int           SetProofServEnv(int psid = -1, int loglevel = -1, const char *cfg = 0);
   int           SetProofServEnvOld(int psid = -1, int loglevel = -1, const char *cfg = 0);
   int           SetShutdownTimer(XrdProofServProxy *xps, bool on = 1);
   int           TerminateProofServ(XrdProofServProxy *xps, bool add = 1);
   int           VerifyProofServ(XrdProofServProxy *xps);

   //
   // Local area
   //
   XrdObject<XrdProofdProtocol>  fProtLink;
   XrdLink                      *fLink;
   XrdBuffer                    *fArgp;
   char                          fStatus;
   char                         *fClientID;    // login username
   char                         *fGroupID;     // login group name
   XrdProofUI                    fUI;           // user info
   unsigned char                 fCapVer;
   kXR_int32                     fSrvType;      // Master or Worker
   bool                          fTopClient;    // External client (not ProofServ)
   bool                          fSuperUser;    // TRUE for privileged clients (admins)
   //
   XrdProofdClient              *fPClient;    // Our reference XrdProofdClient
   kXR_int32                     fCID;        // Reference ID of this client
   //
   XrdSecEntity                 *fClient;
   XrdSecProtocol               *fAuthProt;
   XrdSecEntity                  fEntity;
   //
   char                         *fBuff;
   int                           fBlen;
   int                           fBlast;
   //
   int                           fhcPrev;
   int                           fhcMax;
   int                           fhcNext;
   int                           fhcNow;
   int                           fhalfBSize;
   //
   XPClientRequest               fRequest;  // handle client requests
   XrdProofdResponse             fResponse; // Response to incoming request
   XrdOucRecMutex                fMutex;    // Local mutex

   //
   // Static area: general protocol managing section
   //
   static XrdOucRecMutex         fgXPDMutex;  // Mutex for static area
   static int                    fgCount;
   static XrdObjectQ<XrdProofdProtocol> fgProtStack;
   static XrdBuffManager        *fgBPool;     // Buffer manager
   static int                    fgMaxBuffsz;    // Maximum buffer size we can have
   static XrdSecService         *fgCIA;       // Authentication Server
   static XrdScheduler          *fgSched;     // System scheduler
   static XrdOucError            fgEDest;     // Error message handler

   //
   // Static area: protocol configuration section
   //
   static XrdProofdFile          fgCfgFile;    // Main config file
   static bool                   fgConfigDone; // Whether configure has been run
   static kXR_int32              fgSrvType;    // Master, Submaster, Worker or any
   static std::list<XrdROOT *>   fgROOT;     // ROOT versions; the first is the default
   static char                  *fgTMPdir;   // directory for temporary files
   static char                  *fgImage;    // image name for these servers
   static char                  *fgWorkDir;  // working dir for these servers
   static char                  *fgDataSetDir;  // dataset dir for this master server
   static int                    fgPort;
   static char                  *fgSecLib;
   // 
   static XrdOucString           fgEffectiveUser;  // Effective user
   static XrdOucString           fgLocalHost;  // FQDN of this machine
   static char                  *fgPoolURL;    // Local pool URL
   static char                  *fgNamespace;  // Local pool namespace
   //
   static XrdOucSemWait          fgForkSem;   // To serialize fork requests
   //
   static EResourceType          fgResourceType; // resource type
   static int                    fgMaxSessions; // max number of sessions per client
   static int                    fgMaxOldLogs; // max number of old sessions workdirs per client
   static int                    fgWorkerMax; // max number or workers per user
   static EStaticSelOpt          fgWorkerSel; // selection option
   static std::vector<XrdProofWorker *> fgWorkers;  // vector of possible workers
   static std::list<XrdOucString *> fgMastersAllowed;  // list of master (domains) allowed
   static std::list<XrdProofdPriority *> fgPriorities;  // list of {users, priority change}
   static char                  *fgSuperUsers;  // ':' separated list of privileged users
   //
   static XrdProofdFile          fgPROOFcfg; // PROOF static configuration
   static bool                   fgWorkerUsrCfg; // user cfg files enabled / disabled
   //
   static int                    fgReadWait;
   static int                    fgInternalWait; // Timeout on replies from proofsrv
   //
   static kXR_int32              fgShutdownOpt; // What to do when a client disconnects
   static kXR_int32              fgShutdownDelay; // Delay shutdown by this (if enabled)
   // 
   static int                    fgCron; // Cron thread option [1 ==> start]
   static int                    fgCronFrequency; // Frequency for running cron checks in secs
   //
   static int                    fgOperationMode; // Operation mode
   static XrdOucString           fgAllowedUsers; // Users allowed in controlled mode
   //
   static XrdOucString           fgProofServEnvs; // Additional envs to be exported before proofserv
   static XrdOucString           fgProofServRCs; // Additional rcs to be passed to proofserv
   //
   static int                    fgNumLocalWrks; // Number of local workers [== n cpus]
   //
   static int                    fgNumGroups; // Number of groups

   //
   // Static area: client section
   //
   static std::list<XrdProofdClient *> fgProofdClients;  // keeps track of all users
   static std::list<XrdProofdPInfo *> fgTerminatedProcess; // List of pids of processes terminating

   //
   // Static area: methods
   //
   static int    AddNewSession(XrdProofdClient *client, const char *tag);
   static int    ChangeProcessPriority(int pid, int deltap);
   static int    CheckIf(XrdOucStream *s);
   static bool   CheckMaster(const char *m);
   static int    CheckUser(const char *usr, XrdProofUI &ui, XrdOucString &e);
   static int    Config(const char *fn);
   static int    CreateDefaultPROOFcfg();
   static char  *FilterSecConfig(const char *cfn, int &nd);
   static int    GetNumCPUs();
   static int    GetSessionDirs(XrdProofdClient *pcl, int opt,
                                std::list<XrdOucString *> *sdirs,
                                XrdOucString *tag = 0);
   static int    GetWorkers(XrdOucString &workers, XrdProofServProxy *);
   static int    GuessTag(XrdProofdClient *pcl, XrdOucString &tag, int ridx = 1);
   static XrdSecService *LoadSecurity(char *seclib, char *cfn);
   static int    MvOldSession(XrdProofdClient *client,
                              const char *tag, int maxold = 10);
   static int    ReadPROOFcfg();
   static int    ResolveKeywords(XrdOucString &s, XrdProofdClient *pcl);
   static int    SetProofServEnv(XrdROOT *r);
   static int    SaveAFSkey(XrdSecCredentials *c, const char *fn);
   static int    VerifyProcessByID(int pid, const char *pname = 0);
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofWorker                                                       //
//                                                                      //
// Authors: G. Ganis, CERN, 2006                                        //
//                                                                      //
// Small class with information about a potential worker.               //
// A list of instances of this class is built using the config file or  //
// or the information collected from the resource discoverers.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class XrdProofWorker {

 public:
   XrdProofWorker(const char *str = 0);
   virtual ~XrdProofWorker() { }

   void                    Reset(const char *str); // Set from 'str'

   const char             *Export();

   bool                    Matches(const char *host);

   // Counters
   int                     fActive;      // number of active sessions
   int                     fSuspended;   // number of suspended sessions 

   std::list<XrdProofServProxy *> fProofServs; // ProofServ sessions using
                                               // this worker

   // Worker definitions
   XrdOucString            fExport;      // export string
   char                    fType;        // type: worker ('W') or submaster ('S')
   XrdOucString            fHost;        // user@host
   int                     fPort;        // port
   int                     fPerfIdx;     // performance index
   XrdOucString            fImage;       // image name
   XrdOucString            fWorkDir;     // work directory
   XrdOucString            fMsd;         // mass storage domain
   XrdOucString            fId;          // ID string
};

#endif
