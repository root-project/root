/******************************************************************************/
/*                                                                            */
/*                     X r d C m s P r e p A r g s . c c                      */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// Original Version: 1.2 2007/07/26 15:18:24 ganis

const char *XrdCmsPrepargsCVSID = "$Id$";
  
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>

#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsPrepare.hh"
#include "XrdCms/XrdCmsPrepArgs.hh"

using namespace XrdCms;

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/

XrdSysMutex     XrdCmsPrepArgs::PAQueue;
XrdSysSemaphore XrdCmsPrepArgs::PAReady(0);

XrdCmsPrepArgs *XrdCmsPrepArgs::First = 0;
XrdCmsPrepArgs *XrdCmsPrepArgs::Last  = 0;
int             XrdCmsPrepArgs::isIdle= 1;
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCmsPrepArgs::XrdCmsPrepArgs(XrdCmsRRData &Arg) : XrdJob("prepare")
{

// Copy variable pointers and steal teh data buffer behind them
//
   Request = Arg.Request; Request.streamid = 0;
   Ident   = Arg.Ident;
   reqid   = Arg.Reqid;
   notify  = Arg.Notify;
   prty    = Arg.Prty;
   mode    = Arg.Mode;
   path    = Arg.Path;
   pathlen = Arg.PathLen;
   opaque  = Arg.Opaque;
   options = Arg.Request.modifier;
   Data    = Arg.Buff; Arg.Buff = 0; Arg.Blen = 0;

// Fill out co-location information
//
   if (options & CmsPrepAddRequest::kYR_stage
   &&  options & CmsPrepAddRequest::kYR_coloc && prty)
      {clPath = prty;
       while(*clPath && *clPath != '/') clPath++;
       if (*clPath != '/') clPath = 0;
      } else clPath = 0;

// Fill out the iovec
//
   ioV[0].iov_base = (char *)&Request;
   ioV[0].iov_len  = sizeof(Request);
   ioV[1].iov_base = Data;
   ioV[1].iov_len  = Arg.Dlen;
}

/******************************************************************************/
/*                            g e t R e q u e s t                             */
/******************************************************************************/
  
XrdCmsPrepArgs *XrdCmsPrepArgs::getRequest() // Static
{
   XrdCmsPrepArgs *parg;

// Wait for a request
//
   do {PAQueue.Lock();
       if ((parg = First))
          if (parg == Last) First = Last = 0;
             else           First = parg->Next;
          else {isIdle = 1; PAQueue.UnLock(); PAReady.Wait();}
      } while(parg == 0);
   isIdle = 0;
   PAQueue.UnLock();
   return parg;
}

/******************************************************************************/
/*                               P r o c e s s                                */
/*****************************************************************************/

// This static entry is started on a thread during configuration
//
void XrdCmsPrepArgs::Process()
{
   XrdCmsPrepArgs *aP;

// Process all queued prepare arguments. If we have data then we do this
// for real. Otherwise, simply do a server selection and, if need be, tell 
// the server to stage the file.
//
   if (Config.DiskOK)
      do {aP = getRequest();
          PrepQ.Prepare(aP);
          delete aP;
         } while(1);
      else
      do {getRequest()->DoIt();
         } while(1);
}
  
/******************************************************************************/
/*                                 Q u e u e                                  */
/******************************************************************************/
  
void XrdCmsPrepArgs::Queue()
{

// Lock the queue and add the element and post the waiter
//
   PAQueue.Lock();
   if (First) Last->Next = this;
      else    First      = this;
   Last = this;
   if (isIdle) PAReady.Post();
   PAQueue.UnLock();
}
