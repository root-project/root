// @(#)root/proofd:$Id$
// Author: G. Ganis Jan 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdClientMgr
#define ROOT_XrdProofdClientMgr

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdClientMgr                                                   //
//                                                                      //
// Author: G. Ganis, CERN, 2008                                         //
//                                                                      //
// Class managing clients.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <list>

#include "XpdSysPthread.h"

#include "XrdProofdConfig.h"

#include "XrdOuc/XrdOucString.hh"

#define XPD_LOGGEDIN       1
#define XPD_NEED_AUTH      2
#define XPD_ADMINUSER      4
#define XPD_NEED_MAP       8

class XrdProofdClient;
class XrdProofdConfig;
class XrdProofdManager;
class XrdProofdProtocol;
class XrdProtocol_Config;
class XrdSysError;
class XrdSecService;


class XrdProofdClientMgr : public XrdProofdConfig {

   XrdSysRecMutex    *fMutex;
   XrdProofdManager  *fMgr;
   XrdOucString       fSecLib;
   XrdSecService     *fCIA;            // Authentication Server

   int                fCheckFrequency;
   XrdProofdPipe      fPipe;

   XrdOucString       fClntAdminPath;  // Client admin area
   int                fNDisconnected;  // Clients previously connected still offline
   int                fReconnectTimeOut;
   int                fActivityTimeOut;

   std::list<XrdProofdClient *> fProofdClients;        // keeps track of all users

   int                CheckAdminPath(XrdProofdProtocol *p,
                                     XrdOucString &cidpath, XrdOucString &emsg);
   int                CheckClient(XrdProofdProtocol *p,
                                    const char *user, XrdOucString &emsg);
   int                CreateAdminPath(XrdProofdProtocol *p,
                                      XrdOucString &path, XrdOucString &e);
   int                RestoreAdminPath(XrdOucString &cpath, XrdOucString &emsg);
   int                ParsePreviousClients(XrdOucString &emsg);
   int                MapClient(XrdProofdProtocol *xp, bool all = 1);
   char              *FilterSecConfig(int &nd);

   void               RegisterDirectives();
   int                DoDirectiveClientMgr(char *, XrdOucStream *, bool);

   // Security service
   XrdSecService     *LoadSecurity();

public:
   XrdProofdClientMgr(XrdProofdManager *mgr, XrdProtocol_Config *pi, XrdSysError *e);
   virtual ~XrdProofdClientMgr() { SafeDel(fMutex); }

   enum CMProtocol { kClientDisconnect = 0 };

   int               Config(bool rcf = 0);
   int               DoDirective(XrdProofdDirective *d,
                                 char *val, XrdOucStream *cfg, bool rcf);
   int               CheckClients();

   XrdProofdClient  *GetClient(const char *usr, const char *grp = 0, bool create = 1);
   int               GetNClients() const { XrdSysMutexHelper mh(fMutex);
                                           return fProofdClients.size(); }

   void              Broadcast(XrdProofdClient *c, const char *msg);
   void              TerminateSessions(XrdProofdClient *c, const char *msg, int srvtype);

   int               Process(XrdProofdProtocol *p);

   int               Auth(XrdProofdProtocol *xp);
   int               Login(XrdProofdProtocol *xp);

   int               CheckFrequency() const { return fCheckFrequency; }
   inline XrdProofdPipe *Pipe() { return &fPipe; }
};
#endif
