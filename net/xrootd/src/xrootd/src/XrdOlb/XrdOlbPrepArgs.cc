/******************************************************************************/
/*                                                                            */
/*                     X r d O l b P r e p A r g s . c c                      */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOlbPrepargsCVSID = "$Id$";
  
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>

#include "XrdOlb/XrdOlbConfig.hh"
#include "XrdOlb/XrdOlbPrepArgs.hh"

using namespace XrdOlb;

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/

XrdSysMutex     XrdOlbPrepArgs::PAQueue;
XrdSysSemaphore XrdOlbPrepArgs::PAReady(0);

XrdOlbPrepArgs *XrdOlbPrepArgs::First = 0;
XrdOlbPrepArgs *XrdOlbPrepArgs::Last  = 0;
  
/******************************************************************************/
/*                               p r e p M s g                                */
/******************************************************************************/
  
int XrdOlbPrepArgs::prepMsg()
{

// There is no message unless we are primed to stage
//
   if (!Stage) return 0;

// Create prepare message to be used when a server is selected:
// <mid> prepadd <reqid> <usr> <prty> <mode> <path>\n
//
   Msg[ 0].iov_base = Config.MsgGID;      Msg[ 0].iov_len  = Config.MsgGIDL;
   Msg[ 1].iov_base = (char *)"prepadd "; Msg[ 1].iov_len  = 8;
   Msg[ 2].iov_base = reqid;              Msg[ 2].iov_len  = strlen(reqid);
   Msg[ 3].iov_base = (char *)" ";        Msg[ 3].iov_len  = 1;
   Msg[ 4].iov_base = user;               Msg[ 4].iov_len  = strlen(user);
   Msg[ 5].iov_base = (char *)" ";        Msg[ 5].iov_len  = 1;
   Msg[ 6].iov_base = prty;               Msg[ 6].iov_len  = strlen(prty);
   Msg[ 7].iov_base = (char *)" ";        Msg[ 7].iov_len  = 1;
   Msg[ 8].iov_base = mode;               Msg[ 8].iov_len  = strlen(mode);
   Msg[ 9].iov_base = (char *)" ";        Msg[ 9].iov_len  = 1;
   Msg[10].iov_base = path;               Msg[10].iov_len  = strlen(path);
   Msg[11].iov_base = (char *)"\n";       Msg[11].iov_len  = 1;
   return iovNum;
}

/******************************************************************************/

int XrdOlbPrepArgs::prepMsg(const char *Cmd, const char *Info)
{

// Create message to be sent
//
   Msg[0].iov_base = (char *)Cmd;  Msg[0].iov_len  = strlen(Cmd);
   Msg[1].iov_base = (char *)" ";  Msg[1].iov_len  = 1;
   Msg[2].iov_base = reqid;        Msg[2].iov_len  = strlen(reqid);
   Msg[3].iov_base = (char *)" ";  Msg[3].iov_len  = 1;
   Msg[4].iov_base = (char *)Info; Msg[4].iov_len  = strlen(Info);
   Msg[5].iov_base = (char *)" ";  Msg[5].iov_len  = 1;
   Msg[6].iov_base = path;         Msg[6].iov_len  = strlen(path);
   Msg[7].iov_base = (char *)"\n"; Msg[7].iov_len  = 1;
   return 8;
}

/******************************************************************************/
/*                                 Q u e u e                                  */
/******************************************************************************/
  
void XrdOlbPrepArgs::Queue()
{

// Lock the queue and add the element
//
   PAQueue.Lock();
   if (First) Last->Next = this;
      else    First      = this;
   Last = this;
   PAQueue.UnLock();

// Tell whoever is waiting that we have a ready request
//
   PAReady.Post();
}

/******************************************************************************/
/*                               R e q u e s t                                */
/******************************************************************************/
  
XrdOlbPrepArgs *XrdOlbPrepArgs::Request()
{
   XrdOlbPrepArgs *parg;

// Wait for a request
//
   do {PAReady.Wait();
       PAQueue.Lock();
       if ((parg = First))
          if (parg == Last) First = Last = 0;
             else           First = parg->Next;
       PAQueue.UnLock();
      } while(parg == 0);
   return parg;
}
