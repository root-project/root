// @(#)root/proofd:$Name:  $:$Id: XrdProofdProtocol.h,v 1.4 2006/03/01 15:46:33 rdm Exp $
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
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdNet/XrdNet.hh"

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

class XrdOucError;
class XrdOucTrace;
class XrdBuffer;
class XrdLink;
class XrdProofClient;
class XrdProofWorker;
class XrdScheduler;
class XrdProofdPriority;

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
   int           GetData(const char *dtype, char *buff, int blen);
   int           GetFreeServID();
   XrdProofServProxy *GetServer(int psid);
   int           Interrupt();
   int           Login();
   int           MapClient(bool all = 1);
   int           Ping();
   int           Process2();
   void          Reset();
   int           SendMsg();
   void          SetIgnoreZombieChild();
   void          SetProofServEnv(int psid);
   int           SetUserEnvironment(const char *usr, const char *dir = 0);

   // Static methods
   static int    ChangeProcessPriority(int pid, int deltap);
   static int    CheckIf(XrdOucStream *s);
   static bool   CheckMaster(const char *m);
   static int    Config(const char *fn);
   static char  *Expand(char *p);
   static char  *FilterSecConfig(const char *cfn, int &nd);
   static int    GetWorkers(XrdOucString &workers, XrdProofServProxy *);
   static XrdSecService *LoadSecurity(char *seclib, char *cfn);
   static int    ReadPROOFcfg();
   static int    SetSrvProtVers();

   // Local members
   XrdObject<XrdProofdProtocol>  fProtLink;
   XrdLink                      *fLink;
   XrdBuffer                    *fArgp;
   char                          fStatus;
   char                         *fClientID;
   unsigned char                 fCapVer;
   kXR_int32                     fSrvType;    // Master or Worker
   bool                          fTopClient;  // External client (not ProofServ)
   XrdNet                       *fUNIXSock; // UNIX server socket for internal connections
   char                         *fUNIXSockPath; // UNIX server socket path

   XrdProofClient               *fPClient;    // Our reference XrdProofClient
   kXR_int32                     fCID;        // Reference ID of this client

   XrdSecEntity                 *fClient;
   XrdSecProtocol               *fAuthProt;
   XrdSecEntity                  fEntity;

   // Global members
   static std::list<XrdProofClient *> fgProofClients;  // keeps track of all users
   static int                    fgCount;
   static XrdObjectQ<XrdProofdProtocol> fgProtStack;

   static XrdBuffManager        *fgBPool;     // Buffer manager
   static XrdSecService         *fgCIA;       // Authentication Server
   static bool                   fgConfigDone; // Whether configure has been run

   static XrdScheduler          *fgSched;     // System scheduler
   static XrdOucError            fgEDest;     // Error message handler

   static int                    fgReadWait;
   static int                    fgInternalWait; // Timeout on replies from proofsrv
   static int                    fgPort;
   static char                  *fgSecLib;

   static char                  *fgPrgmSrv;  // PROOF server application
   static int                    fgSrvProtVers;  // Protocol version run by PROOF server
   static char                  *fgROOTsys;  // ROOTSYS
   static char                  *fgTMPdir;   // directory for temporary files
   static char                  *fgImage;    // image name for these servers
   static char                  *fgWorkDir;  // working dir for these servers
   static int                    fgMaxSessions; // max number of sessions per client
   static std::list<XrdOucString *> fgMastersAllowed;  // list of master (domains) allowed
   static std::list<XrdProofdPriority *> fgPriorities;  // list of {users, priority change}
   static kXR_int32              fgSrvType;    // Master, Submaster, Worker or any
   static XrdOucString           fgLocalHost;  // FQDN of this machine
   static char                  *fgPoolURL;    // Local pool URL
   static char                  *fgNamespace;  // Local pool namespace

   static int                    fgMaxBuffsz;    // Maximum buffer size we can have

   static EResourceType          fgResourceType; // resource type

   static char                  *fgPROOFcfg; // PROOF static configuration
   static int                    fgWorkerMax; // max number or workers per user
   static EStaticSelOpt          fgWorkerSel; // selection option
   static bool                   fgWorkerUsrCfg; // user cfg files enabled / disabled
   static std::vector<XrdProofWorker *> fgWorkers;  // vector of possible workers

   static XrdOucMutex            fgXPDMutex;  // Mutex for static area

   char                         *fBuff;
   int                           fBlen;
   int                           fBlast;

   static int                    fghcMax;
   int                           fhcPrev;
   int                           fhcNext;
   int                           fhcNow;
   int                           fhalfBSize;

   XPClientRequest               fRequest; // handle client requests
   XrdProofdResponse             fResponse; // Response to incomign request
   XrdOucMutex                   fMutex; // Local mutex
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
   XrdProofClient(XrdProofdProtocol *p)
                              { fClientID = (p && p->GetID()) ? strdup(p->GetID()) : 0;
                                fProofServs.reserve(10); fClients.reserve(10); }
   virtual ~XrdProofClient()
                              { if (fClientID) free(fClientID); }


   inline const char      *ID() const
                              { return (const char *)fClientID; }
   bool                    Match(const char *id)
                              { return (id ? !strcmp(id, fClientID) : 0); }

   int                     GetClientID(XrdProofdProtocol *p);

   std::vector<XrdProofServProxy *> fProofServs; // Allocated ProofServ sessions
   std::vector<XrdProofdProtocol *> fClients;    // Attached Client sessions

   XrdOucMutex                      fMutex; // Local mutex

 private:
   char                            *fClientID;  // String identifying this client
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
