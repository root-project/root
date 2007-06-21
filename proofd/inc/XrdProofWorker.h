// @(#)root/proofd:$Name:  $:$Id:$
// Author: Gerardo Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofWorker
#define ROOT_XrdProofWorker

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofWorker                                                       //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Class with information about a potential worker.                     //
// A list of instances of this class is built using the config file or  //
// or the information collected from the resource discoverers.          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <list>

#include "XrdOuc/XrdOucString.hh"

class XrdProofServProxy;

class XrdProofWorker {

 public:
   XrdProofWorker(const char *str = 0);
   virtual ~XrdProofWorker() { }

   void                    Reset(const char *str); // Set from 'str'

   const char             *Export();

   bool                    Matches(const char *host);
   int                     GetNActiveSessions();

   static void             Sort(std::list<XrdProofWorker *> *lst,
                                bool (*f)(XrdProofWorker *&lhs,
                                          XrdProofWorker *&rhs));
   // Counters
   int                     fActive;      // number of active sessions
   int                     fSuspended;   // number of suspended sessions 

   std::list<XrdProofServProxy *> fProofServs; // ProofServ sessions using
                                               // this worker

   // Worker definitions
   XrdOucString            fExport;      // export string
   char                    fType;        // type: worker ('W') or submaster ('S')
   XrdOucString            fHost;        // user@host
   int                     fPort;        // port
   int                     fPerfIdx;     // performance index
   XrdOucString            fImage;       // image name
   XrdOucString            fWorkDir;     // work directory
   XrdOucString            fMsd;         // mass storage domain
   XrdOucString            fId;          // ID string
};

#endif
