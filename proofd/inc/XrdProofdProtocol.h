// @(#)root/proofd:$Name:  $:$Id: XrdProofdProtocol.h,v 1.19 2006/12/03 23:34:04 rdm Exp $
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

// User Info class
class XrdProofUI {
public:
   XrdOucString fUser;
   XrdOucString fHomeDir;
   XrdOucString fWorkDir;
   int          fUid;
   int          fGid;

   XrdProofUI() { fUid = -1; fGid = -1; }
   ~XrdProofUI() { }

   void Reset() { fUser = ""; fHomeDir = ""; fWorkDir = ""; fUid = -1; fGid = -1; }
};

class XrdBuffer;
class XrdClientMessage;
class XrdLink;
class XrdOucError;
class XrdOucTrace;
class XrdProofClient;
class XrdProofdPInfo;
class XrdProofdPriority;
class XrdProofWorker;
class XrdScheduler;
class XrdSrvBuffer;

class XrdProofdProtocol : XrdProtocol {

friend class XrdProofClient;

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
   char                         *fClientID;
   XrdProofUI                    fUI;           // user info
   unsigned char                 fCapVer;
   kXR_int32                     fSrvType;      // Master or Worker
   bool                          fTopClient;    // External client (not ProofServ)
   bool                          fSuperUser;    // TRUE for privileged clients (admins)
   //
   XrdProofClient               *fPClient;    // Our reference XrdProofClient
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
   XPClientRequest               fRequest; // handle client requests
   XrdProofdResponse             fResponse; // Response to incoming request
   XrdOucRecMutex                fMutex; // Local mutex

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
   static bool                   fgConfigDone; // Whether configure has been run
   static kXR_int32              fgSrvType;    // Master, Submaster, Worker or any
   static char                  *fgROOTsys;  // ROOTSYS
   static char                  *fgTMPdir;   // directory for temporary files
   static char                  *fgImage;    // image name for these servers
   static char                  *fgWorkDir;  // working dir for these servers
   static int                    fgPort;
   static char                  *fgSecLib;
   // 
   static XrdOucString           fgEffectiveUser;  // Effective user
   static XrdOucString           fgLocalHost;  // FQDN of this machine
   static char                  *fgPoolURL;    // Local pool URL
   static char                  *fgNamespace;  // Local pool namespace
   //
   static char                  *fgPrgmSrv;  // PROOF server application
   static kXR_int16              fgSrvProtVers;  // Protocol version run by PROOF server
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
   static char                  *fgPROOFcfg; // PROOF static configuration
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
   //
   static int                    fgNumLocalWrks; // Number of local workers [== n cpus]

   //
   // Static area: client section
   //
   static std::list<XrdProofClient *> fgProofClients;  // keeps track of all users
   static std::list<XrdProofdPInfo *> fgTerminatedProcess; // List of pids of processes terminating

   //
   // Static area: methods
   //
   static int    AddNewSession(XrdProofClient *client, const char *tag);
   static int    ChangeProcessPriority(int pid, int deltap);
   static int    CheckIf(XrdOucStream *s);
   static bool   CheckMaster(const char *m);
   static int    Config(const char *fn);
   static int    CreateDefaultPROOFcfg();
   static char  *Expand(char *p);
   static char  *FilterSecConfig(const char *cfn, int &nd);
   static int    GetSessionDirs(XrdProofClient *pcl, int opt,
                                std::list<XrdOucString *> *sdirs,
                                XrdOucString *tag = 0);
   static int    GetWorkers(XrdOucString &workers, XrdProofServProxy *);
   static int    GuessTag(XrdProofClient *pcl, XrdOucString &tag, int ridx = 1);
   static XrdSecService *LoadSecurity(char *seclib, char *cfn);
   static int    MvOldSession(XrdProofClient *client,
                              const char *tag, int maxold = 10);
   static int    ReadPROOFcfg();
   static int    SetProofServEnv(XrdProofdProtocol *p = 0, int psid = -1,
                                 int loglevel = -1, const char *cfg = 0);
   static int    SaveAFSkey(XrdSecCredentials *c, const char *fn);
   static int    SetSrvProtVers();
   static int    ResolveKeywords(XrdOucString &s, XrdProofClient *pcl);
   static int    VerifyProcessByID(int pid, const char *pname = 0);

   static int    GetNumCPUs();
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofClient                                                       //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Small class to map a client in static area.                          //
// When a new client gets in a matching XrdProofClient is searched for. //
// If it is found, the client attachs to it, mapping its content.       //
// If no matching XrdProofClient is found, a new one is created.        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class XrdProofClient {

 public:
   XrdProofClient(XrdProofdProtocol *p,
                  short int clientvers = -1, const char *wrk = 0);

   virtual ~XrdProofClient();

   inline const char      *ID() const
                              { return (const char *)fClientID; }
   bool                    Match(const char *id)
                              { return (id ? !strcmp(id, fClientID) : 0); }
   inline XrdOucRecMutex  *Mutex() const { return (XrdOucRecMutex *)&fMutex; }
   inline unsigned short   RefSid() const { return fRefSid; }
   inline short            Version() const { return fClientVers; }
   inline const char      *Workdir() const
                              { return (const char *)fWorkdir; }
   inline std::vector<XrdProofServProxy *> *ProofServs()
                           { return (std::vector<XrdProofServProxy *> *)&fProofServs; }
   inline std::vector<XrdProofdProtocol *> *Clients()
                           { return (std::vector<XrdProofdProtocol *> *)&fClients; }
   void                    ResetClient(int i) { fClients[i] = 0; }

   void                    EraseServer(int psid);
   int                     GetClientID(XrdProofdProtocol *p);
   int                     GetFreeServID();

   void                    SetRefSid(unsigned short sid) { fRefSid = sid; }
   void                    SetWorkdir(const char *wrk)
                              { if (fWorkdir) free(fWorkdir);
                                fWorkdir = (wrk) ? strdup(wrk) : 0; }

   int                     CreateUNIXSock(XrdOucError *edest, char *tmpdir);
   XrdNet                 *UNIXSock() const { return fUNIXSock; }
   char                   *UNIXSockPath() const { return fUNIXSockPath; }
   void                    SaveUNIXPath(); // Save path in the sandbox
   void                    SetUNIXSockSaved() { fUNIXSockSaved = 1;}

 private:

   XrdOucRecMutex          fMutex; // Local mutex

   char                   *fClientID;   // String identifying this client
   short int               fClientVers; // PROOF version run by client
   unsigned short          fRefSid;     // Reference stream ID for this client
   char                   *fWorkdir;    // Client working area (sandbox) 

   XrdNet                 *fUNIXSock;     // UNIX server socket for internal connections
   char                   *fUNIXSockPath; // UNIX server socket path
   bool                    fUNIXSockSaved; // TRUE if the socket path has been saved

   std::vector<XrdProofServProxy *> fProofServs; // Allocated ProofServ sessions
   std::vector<XrdProofdProtocol *> fClients;    // Attached Client sessions
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdPriority                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2006                                        //
//                                                                      //
// Small class to describe priority changes.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class XrdProofdPriority {
public:
   XrdOucString            fUser;          // User to who this applies (wild cards accepted)
   int                     fDeltaPriority; // Priority change
   XrdProofdPriority(const char *usr, int dp) : fUser(usr), fDeltaPriority(dp) { }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdPInfo                                                       //
//                                                                      //
// Authors: G. Ganis, CERN, 2006                                        //
//                                                                      //
// Small class to describe a process.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class XrdProofdPInfo {
public:
   int pid;
   XrdOucString pname;
   XrdProofdPInfo(int i, const char *n) : pid(i) { pname = n; }
};

#endif
