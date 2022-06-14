// @(#)root/proofd:$Id$
// Author: G. Ganis Feb 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdPriorityMgr
#define ROOT_XrdProofdPriorityMgr

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdPriorityMgr                                                 //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Class managing session priorities.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <list>

#include "XpdSysPthread.h"

#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"

#include "XrdProofdConfig.h"

class XrdProofdDirective;
class XrdProofdManager;

//
// Small class to describe an active session
//
#define XPPM_NOPRIORITY 999999
class XrdProofdSessionEntry {
public:
   XrdOucString fUser;
   XrdOucString fGroup;
   int          fPid;
   int          fDefaultPriority;
   int          fPriority;
   float        fFracEff;  // Resource fraction in % (effective)

   XrdProofdSessionEntry(const char *u, const char *g, int pid);
   virtual ~XrdProofdSessionEntry();

   int          SetPriority(int priority = XPPM_NOPRIORITY);
};

class XrdProofdPriorityMgr : public XrdProofdConfig {

   XrdSysRecMutex    fMutex;          // Atomize this instance

   XrdProofdManager *fMgr;
   XrdOucHash<XrdProofdSessionEntry> fSessions; // list of active sessions (keyed by "pid")
   XrdOucHash<XrdProofdPriority> fPriorities; // list of {users, priority change}

   XrdProofdPipe     fPipe;

   int               fSchedOpt;       // Worker sched option
   int               fPriorityMax;    // Max session priority [1,40], [20]
   int               fPriorityMin;    // Min session priority [1,40], [1]

   void              RegisterDirectives();
   int               DoDirectivePriority(char *, XrdOucStream *, bool);
   int               DoDirectiveSchedOpt(char *, XrdOucStream *, bool);

public:
   XrdProofdPriorityMgr(XrdProofdManager *mgr, XrdProtocol_Config *pi, XrdSysError *e);
   virtual ~XrdProofdPriorityMgr() { }

   enum PMProtocol { kChangeStatus = 0, kSetGroupPriority = 1 };

   int               Config(bool rcf = 0);
   int               DoDirective(XrdProofdDirective *d,
                                 char *val, XrdOucStream *cfg, bool rcf);

   inline XrdProofdPipe *Pipe() { return &fPipe; }

   int               AddSession(const char *u, const char *g, int pid);
   int               RemoveSession(int pid);
   void              SetGroupPriority(const char *grp, int priority);

   // Scheduling
   bool              IsSchedOn() { return ((fSchedOpt != kXPD_sched_off) ? 1 : 0); }
   int               SetNiceValues(int opt = 0);
   void              SetSchedOpt(int opt) { XrdSysMutexHelper mhp(&fMutex);
                                            fSchedOpt = opt; }
   int               SetProcessPriority(int pid, const char *usr, int &dp);
};

#endif
