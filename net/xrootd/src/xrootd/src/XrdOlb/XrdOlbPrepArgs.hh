#ifndef __OLB_PREPARGS__H
#define __OLB_PREPARGS__H
/******************************************************************************/
/*                                                                            */
/*                     X r d O l b P r e p A r g s . h h                      */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$
  
#include "Xrd/XrdJob.hh"
#include "XrdOlb/XrdOlbServer.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdOlbPrepArgs : public XrdJob
{
public:
static const int       iovNum = 12;

       char           *reqid;
       char           *user;
       char           *prty;
       char           *mode;
       char           *path;
       char           *data;
       struct iovec    Msg[iovNum];
       int             Stage;
       int             endP;

       void            Clear() {reqid = user = prty = mode = path = data = 0;
                                Next = 0; Stage = 0;}

       void            DoIt() {if (!XrdOlbServer::Resume(this)) delete this;}

       int             prepMsg();

       int             prepMsg(const char *Cmd, const char *Info);

       void            Queue();

static XrdOlbPrepArgs *Request();

                       XrdOlbPrepArgs(int srvr) : XrdJob("prepare") 
                                        {Clear(); endP = srvr;}

                      ~XrdOlbPrepArgs() {if (data) free(data);}

private:

static XrdSysMutex     PAQueue;
static XrdSysSemaphore PAReady;
       XrdOlbPrepArgs *Next;
static XrdOlbPrepArgs *First;
static XrdOlbPrepArgs *Last;

};
#endif
