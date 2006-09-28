// @(#)root/proofd:$Name:  $:$Id: XrdProofdProtocol.h,v 1.12 2006/08/05 20:04:47 brun Exp $
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
#define XPROOFD_VERSBIN 0x000003E9
#define XPROOFD_VERSION "0.1"

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
   int          fUid;
   int          fGid;

   XrdProofUI() { fUid = -1; fGid = -1; }
   ~XrdProofUI() { }

   void Reset() { fUser = ""; fHomeDir = ""; fUid = -1; fGid = -1; }
};

class XrdBuffer;
class XrdClientMessage;
class XrdLink;
class XrdOucError;
class XrdOucTrace;
class XrdProofClient;
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

 private:

   int           Admin();
   int           Attach();
   int           Auth();
   int           Create();
   int           Destroy();
   int           Detach();
   void          EraseServer(int psid);
   int           GetBuff(int quantum);
   int           GetData(const char *dtype, char *buff, int blen);
   int           GetFreeServID();
   XrdProofServProxy *GetServer(int psid);
   int           Interrupt();
   int           Login();
   int           MapClient(bool all = 1);
   int           Ping();
   int           Process2();
   void          Reset();
   int           SendData(XrdProofdResponse *resp, kXR_int32 sid = -1, XrdSrvBuffer **buf = 0);
   int           SendDataN(XrdProofServProxy *xps, XrdSrvBuffer **buf = 0);
   int           SendMsg();
   int           SetUserEnvironment(XrdProofServProxy *xps, const char *usr, const char *dir = 0);
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
   int           VerifyProcessByID(int pid, const char *pname = 0);

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
   XrdProofdResponse             fResponse; // Response to incomign request
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
   static int                    fgSrvProtVers;  // Protocol version run by PROOF server
   static XrdOucSemWait          fgForkSem;   // To serialize fork requests
   //
   static EResourceType          fgResourceType; // resource type
   static int                    fgMaxSessions; // max number of sessions per client
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
   // Static area: client section
   //
   static std::list<XrdProofClient *> fgProofClients;  // keeps track of all users
   static std::list<int *>       fgTerminatedProcess; // List of pids of processes terminating

   //
   // Static area: methods
   //
   static int    ChangeProcessPriority(int pid, int deltap);
   static int    CheckIf(XrdOucStream *s);
   static bool   CheckMaster(const char *m);
   static int    Config(const char *fn);
   static char  *Expand(char *p);
   static char  *FilterSecConfig(const char *cfn, int &nd);
   static int    GetWorkers(XrdOucString &workers, XrdProofServProxy *);
   static XrdSecService *LoadSecurity(char *seclib, char *cfn);
   static int    ReadPROOFcfg();
   static void   SetIgnoreZombieChild();
   static int    SetProofServEnv(XrdProofdProtocol *p = 0, int psid = -1,
                                 int loglevel = -1, const char *cfg = 0);
   static int    SetSrvProtVers();
   static int    VerifyPID(int pid);
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
   XrdProofClient(XrdProofdProtocol *p, short int clientvers = -1,
                  const char *tag = 0, const char *ord = 0);

   virtual ~XrdProofClient();

   inline const char      *ID() const
                              { return (const char *)fClientID; }
   bool                    Match(const char *id)
                              { return (id ? !strcmp(id, fClientID) : 0); }
   inline unsigned short   RefSid() const { return fRefSid; }
   inline const char      *SessionTag() const
                              { return (const char *)fSessionTag; }
   inline const char      *Ordinal() const
                              { return (const char *)fOrdinal; }
   inline short            Version() const { return fClientVers; }

   int                     GetClientID(XrdProofdProtocol *p);

   void                    SetRefSid(unsigned short sid) { fRefSid = sid; }
   void                    SetSessionTag(const char *tag)
                              { if (fSessionTag) free(fSessionTag);
                                fSessionTag = (tag) ? strdup(tag) : 0; }
   void                    SetOrdinal(const char *ord)
                              { if (fOrdinal) free(fOrdinal);
                                fOrdinal = (ord) ? strdup(ord) : 0; }

   std::vector<XrdProofServProxy *> fProofServs; // Allocated ProofServ sessions
   std::vector<XrdProofdProtocol *> fClients;    // Attached Client sessions

   XrdOucMutex                      fMutex; // Local mutex

 private:
   char                            *fClientID;   // String identifying this client
   char                            *fSessionTag; // [workers, submasters] session tag of the master
   char                            *fOrdinal;    // [workers, submasters] ordinal number 
   short int                        fClientVers; // PROOF version run by client
   unsigned short                   fRefSid;     // Reference stream ID for this client
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

   // Counters
   int                     fActive;      // number of active sessions
   int                     fSuspended;   // number of suspended sessions 

   std::list<XrdProofServProxy *> fProofServs; // ProofServ sessions using
                                               // this worker

   // Worker definitions
   XrdOucString            fExport;    // export string
   char                    fType;        // type: worker ('W') or submaster ('S')
   XrdOucString            fHost;    // user@host
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

#endif
