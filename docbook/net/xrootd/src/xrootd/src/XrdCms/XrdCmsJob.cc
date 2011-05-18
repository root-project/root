/******************************************************************************/
/*                                                                            */
/*                          X r d C m s J o b . c c                           */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

const char *XrdCmsJobCVSID = "$Id$";
 
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>

#include "Xrd/XrdLink.hh"
#include "Xrd/XrdScheduler.hh"

#include "XrdSys/XrdSysHeaders.hh"

#include "XrdCms/XrdCmsJob.hh"
#include "XrdCms/XrdCmsProtocol.hh"
#include "XrdCms/XrdCmsRRData.hh"
#include "XrdCms/XrdCmsTrace.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

namespace XrdCms
{
extern XrdScheduler               *Sched;
};

       XrdSysMutex      XrdCmsJob::JobMutex;
       XrdCmsJob       *XrdCmsJob::JobStack = 0;

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/

XrdCmsJob *XrdCmsJob::Alloc(XrdCmsProtocol *Proto, XrdCmsRRData *Data)
{
   XrdCmsJob *jp;

// Grab a protocol object and, if none, return a new one
//
   JobMutex.Lock();
   if ((jp = JobStack)) JobStack = jp->JobLink;
      else jp = new XrdCmsJob();
   JobMutex.UnLock();

// Copy relevant sections to the newly allocated protocol object
//
   if (jp)
      {jp->theProto = Proto;
       jp->theData  = Data;
       jp->Comment  = Proto->myRole;
       Proto->Link->setRef(1);
      } else Say.Emsg("Job","No more job objects to serve",Proto->Link->Name());

// All done
//
   return jp;
}

/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/
  
void XrdCmsJob::DoIt()
{
   int rc;

// Simply execute the method on the data. If operation started and we have to
// wait foir it, simply reschedule ourselves for a later time.
//
   if ((rc = theProto->Execute(*theData)))
      if (rc == -EINPROGRESS)
         {Sched->Schedule((XrdJob *)this, theData->waitVal+time(0)); return;}
   Recycle();
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdCmsJob::Recycle()
{

// Dereference the link at this point
//
   theProto->Link->setRef(-1);

// Release the data buffer
//
   if (theData) {XrdCmsRRData::Objectify(theData); theData = 0;}

// Push ourselves on the stack
//
   JobMutex.Lock();
   JobLink  = JobStack;
   JobStack = this;
   JobMutex.UnLock();
}
