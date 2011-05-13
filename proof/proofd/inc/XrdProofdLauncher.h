// @(#)root/proofd:$Id$
// Author: G. Ganis March 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdLauncher
#define ROOT_XrdProofdLauncher

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdLauncher                                                    //
//                                                                      //
// Author: G. Ganis, CERN, 2011                                         //
//                                                                      //
// Class describing the proofserv launcher interface                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdOuc/XrdOucString.hh"
#include "XrdProofdClient.h"

class XrdNetPeer;

//
// Structure for inputs
typedef struct {
   XrdProofdManager   *fMgr;  // General Xproof manager
   XrdProofdProofServ *fPS;   // Object describing the session
   int          fDbgLevel;    // Debug level
   XrdOucString fEnvFile;     // File describing the session environment
   XrdOucString fSessionDir;  // Template for the session directory path
   XrdOucString fErrLog;      // File for startup error logging
   int          fIntWait;     // Timeout on internal connections
   void        *fAux;         // Container for extensions
} ProofdLaunch_t;

class rpdunix;

class XrdProofdLauncher {

 public:
   XrdProofdLauncher(XrdProofdClient *c) : fClient(c) { }
   virtual ~XrdProofdLauncher() { }

   //
   // To check if the instance is valid
   virtual bool Valid() { return (fClient->IsValid()) ? 1 : 0; }
   
   //
   // Launch the session, establishing the UNIX connection and
   // retrieve the process id; returns the peer object describing the connection
   // and the pid. Or NULL in case of failure.
   virtual XrdNetPeer *Launch(ProofdLaunch_t *in,  // Object describing inputs
                              int &pid);           // ID of the process started 

   //
   // Called before Launch, for optional pre-actions
   virtual void Pre() { }

   //
   // Called after Launch, for optional post-actions
   virtual void Post() { }

 protected:

   //
   // Setup the connected peer (called by Launch(...))
   XrdNetPeer *SetupPeer(ProofdLaunch_t *in, int &pid, rpdunix *uconn);

   XrdProofdClient  *fClient;      // Owner
};

#endif
