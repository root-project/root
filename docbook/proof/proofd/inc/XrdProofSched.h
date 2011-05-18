// @(#)root/proofd:$Id$
// Author: G. Ganis  Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofSched
#define ROOT_XrdProofSched

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofSched                                                        //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Interface for a PROOF scheduler.                                     //
// Alternative scheduler implementations should be provided as shared   //
// library containing an implementation of this class. The library      //
// must also provide a function to load an instance of this class       //
// with the following signature (see commented example in               //
// XrdProofSched.cxx):                                                  //
// extern "C" {                                                         //
//    XrdProofSched *XrdgetProofSched(const char *cfg,                  //
//                                    XrdProofdManager *mgr,            //
//                                    XrdProofGroupMgr *grpmgr,         //
//                                    XrdSysError *edest);              //
// }                                                                    //
// Here 'cfg' is the xrootd config file where directives to configure   //
// the scheduler are specified, 'mgr' is the instance of the cluster    //
// manager from where the scheduler can get info about the available    //
// workers and their status, 'grpmgr' is the instance of the group      //
// bringing the definition of the groups for this run, and 'edest' is   //
// instance of the error logger to be used.                             //
// The scheduler is identified by a name of max 16 chars.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <list>

#include "XrdProofdAux.h"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"
#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#endif

#include "XrdProofdConfig.h"

#define kXPSMXNMLEN 17

class XrdProofdManager;
class XrdProofGroupMgr;
class XrdProofdProofServ;
class XrdProofWorker;
class XrdSysError;
class XrdOucStream;

class XrdProofSched : public XrdProofdConfig {

public:
   XrdProofSched(const char *name,
                 XrdProofdManager *mgr, XrdProofGroupMgr *grpmgr,
                 const char *cfn, XrdSysError *e = 0);
   virtual ~XrdProofSched() { }

   // Returns list of workers to be used by session 'xps'.
   // The return code must be one of the following:
   //  -1     Some failure occured; cannot continue
   //   0     A new list has been assigned to the session 'xps' and
   //         returned in 'wrks'
   //   1     The list currently assigned to the session is the one
   //         to be used
   //   2     No worker could be assigned now; session should be queued
   virtual int GetWorkers(XrdProofdProofServ *xps,
                          std::list<XrdProofWorker *> */*wrks*/,
                          const char *);

   // To be called after some nodes become free
   virtual int Reschedule();

   // Update info about a session
   virtual int UpdateSession(XrdProofdProofServ *, int = 0, void * = 0) { return 0; }

   // Max number of essions we are allowed to start
   virtual int MaxSessions() const { return fMaxSessions; }

   // Update group properties according to the current state
   virtual int UpdateProperties() { return 0; }

   virtual int ExportInfo(XrdOucString &);

   virtual bool IsValid() { return fValid; }

   const char *Name() const { return (const char *) &fName[0]; }

   virtual int ProcessDirective(XrdProofdDirective *d,
                                char *val, XrdOucStream *cfg, bool rcf);
   virtual int Enqueue(XrdProofdProofServ *xps, XrdProofQuery *query);
   virtual void DumpQueues(const char *prefix = 0);

   virtual XrdProofdProofServ *FirstSession();

   int         CheckFrequency() const { return fCheckFrequency; }
   inline XrdProofdPipe *Pipe() { return &fPipe; }

   virtual int Config(bool rcf = 0);
   virtual int DoDirective(XrdProofdDirective *d,
                                 char *val, XrdOucStream *cfg, bool rcf);

   enum SchedProtocol { kReschedule = 0 };

protected:
   char              fName[kXPSMXNMLEN];   // Name of this protocol
   bool              fValid;  // TRUE if the scheduler is usable
   XrdProofdManager *fMgr;    // Cluster manager
   XrdProofGroupMgr *fGrpMgr;  // Groups manager

   int               fMaxSessions; // max number of sessions
   int               fMaxRunning;  // max number of running sessions
   int               fWorkerMax;   // max number or workers per user
   int               fWorkerSel;   // selection option
   int               fNextWrk;     // Reference index for RR sel option
   int               fOptWrksPerUnit; // optimal # of workers per CPU/HD
   int               fMinForQuery; // Minimal number of workers for a query
   double            fNodesFraction; // the fraction of free units to assign
                                     // to a query.
   bool              fUseFIFO;    // use FIFO or refuse if overloaded 
   std::list<XrdProofdProofServ *> fQueue; // the queue with sessions (jobs);

   XrdOucHash<XrdProofdDirective> fConfigDirectives; // Config directives

   int               fCheckFrequency;
   XrdProofdPipe     fPipe;

   XrdSysError      *fEDest;      // Error message handler


   virtual void      RegisterDirectives();
   virtual int       DoDirectiveSchedParam(char *, XrdOucStream *, bool);
   virtual int       DoDirectiveResource(char *, XrdOucStream *, bool);

   virtual int       GetNumWorkers(XrdProofdProofServ *xps);
   virtual void      ResetParameters();
};


// Plugin loader handle
typedef XrdProofSched *(*XrdProofSchedLoader_t)(const char *, XrdProofdManager *,
                                                XrdProofGroupMgr *, const char *,
                                                XrdSysError *);

#endif
