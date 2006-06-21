// @(#)root/proofd:$Name:  $:$Id: XrdProofServProxy.cxx,v 1.5 2006/04/19 10:57:44 rdm Exp $
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string.h>
#include <unistd.h>
#include <sys/uio.h>

#include <list>
#include <map>

#include "XrdProofServProxy.h"
#include "XrdProofdProtocol.h"

// Tracing utils
#include "XrdProofdTrace.h"
extern XrdOucTrace *XrdProofdTrace;
static const char *TraceID = " ";
#define TRACEID TraceID
#ifndef SafeDelete
#define SafeDelete(x) { if (x) { delete x; x = 0; } }
#endif

//__________________________________________________________________________
XrdProofServProxy::XrdProofServProxy()
{
   // Constructor

   fLink = 0;
   fParent = 0;
   fPingSem = 0;
   fQueryNum = 0;
   fStartMsg = 0;
   fStatus = kXPD_idle;
   fSrvID = -1;
   fSrvType = kXPD_TopMaster;
   fID = -1;
   fIsValid = true;  // It is created for a valid server ...
   fProtVer = -1;
   strcpy(fFileout,"none");
   memset(fClient, 0, 9);
   strcpy(fTag,"");
   strcpy(fAlias,"");
   fClients.reserve(10);
}

//__________________________________________________________________________
XrdProofServProxy::~XrdProofServProxy()
{
   // Destructor

   SafeDelete(fQueryNum);
   SafeDelete(fStartMsg);
   SafeDelete(fPingSem);

   std::vector<XrdClientID *>::iterator i;
   for (i = fClients.begin(); i != fClients.end(); i++)
       if (*i)
          delete (*i);
   fClients.clear();

   // Cleanup worker info
   ClearWorkers();
}

//__________________________________________________________________________
void XrdProofServProxy::ClearWorkers()
{
   // Decrease worker counters and clean-up the list

   // Decrease worker counters
   std::list<XrdProofWorker *>::iterator i;
   for (i = fWorkers.begin(); i != fWorkers.end(); i++)
       if (*i)
          (*i)->fActive--;
   fWorkers.clear();
}

//__________________________________________________________________________
void XrdProofServProxy::Reset()
{
   // Reset this instance
   XrdOucMutexHelper mtxh(&fMutex);

   fLink = 0;
   fParent = 0;
   SafeDelete(fQueryNum);
   SafeDelete(fStartMsg);
   SafeDelete(fPingSem);
   fStatus = kXPD_idle;
   fSrvID = -1;
   fSrvType = kXPD_TopMaster;
   fID = -1;
   fIsValid = 0;
   fProtVer = -1;
   strcpy(fFileout,"none");
   strcpy(fTag,"");
   strcpy(fAlias,"");
   fClients.clear();
   // Cleanup worker info
   ClearWorkers();
}

//__________________________________________________________________________
XrdClientID *XrdProofServProxy::GetClientID(int cid)
{
   // Get instance corresponding to cid
   //

   XrdClientID *csid = 0;

   if (cid < 0) {
      TRACE(ALL,"XrdProofServProxy::GetClientID: negative ID: protocol error!");
      return csid;
   }
   TRACE(ALL,"XrdProofServProxy::GetClientID: cid = "<<cid<<
             "; size = "<<fClients.size());

   // If in the allocate range reset the corresponding instance and
   // return it
   if (cid < (int)fClients.size()) {
      csid = fClients.at(cid);
      csid->Reset();
      return csid;
   }

   // If not, allocate a new one; we need to resize (double it)
   if (cid >= (int)fClients.capacity())
      fClients.reserve(2*fClients.capacity());

   // Allocate new elements (for fast access we need all of them)
   int ic = (int)fClients.size();
   for (; ic <= cid; ic++)
      fClients.push_back((csid = new XrdClientID()));

   TRACE(ALL,"XrdProofServProxy::GetClientID: cid = "<<cid<<
             "; new size = "<<fClients.size());

   // We are done
   return csid;
}

//__________________________________________________________________________
int XrdProofServProxy::GetFreeID()
{
   // Get next free client ID. If none is found, increase the vector size
   // and get the first new one

   int ic = 0;
   // Search for free places in the existing vector
   for (ic = 0; ic < (int)fClients.size() ; ic++) {
      if (fClients[ic] && (fClients[ic]->fP == 0))
         return ic;
   }

   // We need to resize (double it)
   if (ic >= (int)fClients.capacity())
      fClients.reserve(2*fClients.capacity());

   // Allocate new element
   fClients.push_back(new XrdClientID());

   // We are done
   return ic;
}

//__________________________________________________________________________
int XrdProofServProxy::GetNClients()
{
   // Get number of attached clients.

   int nc = 0;
   // Search for free places in the existing vector
   int ic = 0;
   for (ic = 0; ic < (int)fClients.size() ; ic++)
      if (fClients[ic] && fClients[ic]->IsValid())
         nc++;

   // We are done
   return nc;
}

//__________________________________________________________________________
const char *XrdProofServProxy::StatusAsString() const
{
   // Return a string describing the status

   const char *sst[] = { "idle", "running", "shutting-down", "unknown" };

   // Check status range
   int ist = fStatus;
   ist = (ist > kXPD_unknown) ? kXPD_unknown : ist;
   ist = (ist < kXPD_idle) ? kXPD_unknown : ist;

   // Done
   return sst[ist];
}
