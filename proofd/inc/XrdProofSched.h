// @(#)root/proofd:$Name:  $:$Id$
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
//                                    XrdOucError *edest);              //
// }                                                                    //
// Here 'cfg' is the xrootd config file where directives to configure   //
// the scheduler an be specified, 'mgr' is the instance of the cluster  //
// manager from where the scheduler can get info about the available    //
// workers and their status, 'grpmgr' is the instance of the group      //
// bringing the definition of the groups for this run, and 'edest' is   //
// instance of the error logger to be used.                             //
// The scheduler is identified by a name of max 16 chars.               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <list>

#include "XrdOuc/XrdOucString.hh"

#define kXPSMXNMLEN 17

class XrdProofdManager;
class XrdProofGroupMgr;
class XrdProofServProxy;
class XrdProofWorker;
class XrdOucError;

class XrdProofSched
{
public:
   XrdProofSched(const char *name,
                 XrdProofdManager *mgr, XrdProofGroupMgr *grpmgr,
                 const char *cfn = 0, XrdOucError *e = 0);
   virtual ~XrdProofSched() { }

   // Returns list of workers to be used by session 'xps'
   virtual int GetWorkers(XrdProofServProxy *xps,
                          std::list<XrdProofWorker *> */*wrks*/);

   // Update group properties according to the current state
   virtual int UpdateProperties() { return 0; }

   virtual int ExportInfo(XrdOucString &);

   virtual bool IsValid() { return fValid; }

   const char *Name() const { return (const char *) &fName[0]; }

protected:
   char              fName[kXPSMXNMLEN];   // Name of this protocol
   bool              fValid;  // TRUE if the scheduler is usable
   XrdProofdManager *fMgr;    // Cluster manager
   XrdProofGroupMgr *fGrpMgr;  // Groups manager

   int               fMaxSessions; // max number of sessions per client
   int               fWorkerMax;   // max number or workers per user
   int               fWorkerSel;   // selection option
   int               fNextWrk;     // Reference index for RR sel option 

   XrdOucError      *fEDest;      // Error message handler

   virtual int       Config(const char *cfn);
   virtual int       GetNumWorkers(XrdProofServProxy *xps);
};


// Plugin loader handle
typedef XrdProofSched *(*XrdProofSchedLoader_t)(const char *, XrdProofdManager *,
                                                XrdProofGroupMgr *, XrdOucError *);

#endif
