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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <list>
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"

#include "XrdProofdConfig.h"

class XrdProtocol_Config;
class XrdSysError;
class XrdProofdManager;
class XrdProofdProtocol;
class XrdProofdResponse;
class XpdAdminCpCmd {
public:
   XrdOucString  fCmd;
   XrdOucString  fFmt;
   bool          fCanPut;
   XpdAdminCpCmd(const char *cmd, const char *fmt, bool put) :
                                  fCmd(cmd), fFmt(fmt), fCanPut(put) { }
};

class XrdProofdAdmin : public XrdProofdConfig {

   XrdProofdManager *fMgr;
   std::list<XrdOucString> fExportPaths;
   XrdOucHash<XpdAdminCpCmd> fAllowedCpCmds; // List of copy commands
   XrdOucString      fCpCmds; // String with the allowed copy commands

   void              RegisterDirectives();
   int               DoDirectiveExportPath(char *, XrdOucStream *, bool);
   int               DoDirectiveCpCmd(char *, XrdOucStream *, bool);

   int               CheckForbiddenChars(const char *s);
   int               CheckPath(bool superuser, const char *sbdir, XrdOucString &fullpath,
                               int check, bool &sandbox, struct stat *st, XrdOucString &emsg);
   int               ExecCmd(XrdProofdProtocol *p, XrdProofdResponse *r,
                             int action, const char *cmd, XrdOucString &emsg);
   int               Send(XrdProofdResponse *r, const char *msg);

public:
   XrdProofdAdmin(XrdProofdManager *mgr, XrdProtocol_Config *pi, XrdSysError *e);
   virtual ~XrdProofdAdmin() { }

   int               Config(bool rcf = 0);
   int               DoDirective(XrdProofdDirective *d,
                                 char *val, XrdOucStream *cfg, bool rcf);

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
   int               Exec(XrdProofdProtocol *p);
   int               GetFile(XrdProofdProtocol *p);
   int               PutFile(XrdProofdProtocol *p);
   int               CpFile(XrdProofdProtocol *p);
};

#endif
