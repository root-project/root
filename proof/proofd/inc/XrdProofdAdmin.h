// @(#)root/proofd:$Id$
// Author: G. Ganis Feb 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdAdmin
#define ROOT_XrdProofdAdmin

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdAdmin                                                       //
//                                                                      //
// Author: G. Ganis, CERN, 2008                                         //
//                                                                      //
// Envelop class for admin services.                                    //
// Loaded as service by XrdProofdManager.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class XrdProofdManager;
class XrdProofdProtocol;

class XrdProofdAdmin {

   XrdProofdManager *fMgr;

public:
   XrdProofdAdmin(XrdProofdManager *mgr);
   virtual ~XrdProofdAdmin() { }

   int               Process(XrdProofdProtocol *p, int type);

   int               QuerySessions(XrdProofdProtocol *p);
   int               QueryLogPaths(XrdProofdProtocol *p);
   int               CleanupSessions(XrdProofdProtocol *p);
   int               SendMsgToUser(XrdProofdProtocol *p);
   int               SetGroupProperties(XrdProofdProtocol *p);
   int               GetWorkers(XrdProofdProtocol *p);
   int               QueryWorkers(XrdProofdProtocol *p);
   int               QueryROOTVersions(XrdProofdProtocol *p);
   int               SetROOTVersion(XrdProofdProtocol *p);
   int               SetSessionAlias(XrdProofdProtocol *p);
   int               SetSessionTag(XrdProofdProtocol *p);
   int               ReleaseWorker(XrdProofdProtocol *p);
};

#endif
