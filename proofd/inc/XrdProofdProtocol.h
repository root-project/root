// @(#)root/proofd:$Name:  $:$Id: proofdp.h,v 1.4 2003/08/29 10:41:28 rdm Exp $
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
#define XPROOFD_VERSION "1001"

#define XPD_LOGGEDIN       1
#define XPD_NEED_AUTH      2
#define XPD_ADMINUSER      4
#define XPD_NEED_MAP       8

class XrdOucError;
class XrdOucTrace;
class XrdBuffer;
class XrdLink;
class XrdProofClient;
class XrdScheduler;

class XrdProofdProtocol : XrdProtocol {

friend class XrdProofClient;

public:
   XrdProofdProtocol(const char *rsys = 0, int iw = -1 );
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
   static int    Config(const char *fn);
   static char  *FilterSecConfig(const char *cfn, int &nd);
   static XrdSecService *LoadSecurity(char *seclib, char *cfn);
   static int    Xsecl(XrdOucStream &Config);

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

   static XrdScheduler          *fgSched;     // System scheduler
   static XrdOucError            fgEDest;     // Error message handler

   static int                    fgReadWait;
   static int                    fgInternalWait; // Timeout on replies from proofsrv
   static int                    fgPort;
   static char                  *fgSecLib;

   static char                  *fgPrgmSrv;  // Server application
   static char                  *fgROOTsys;  // ROOTSYS
   static char                  *fgTMPdir;  // directory for temporary files

   static int                    fgMaxBuffsz;    // Maximum buffer size we can have

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

#endif
