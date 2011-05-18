#ifndef __FRMXFRQUEUE_H__
#define __FRMXFRQUEUE_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d F r m X f r Q u e u e . h h                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XrdFrm/XrdFrmRequest.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"

class  XrdFrmReqFile;
class  XrdFrmRequest;
class  XrdFrmXfrJob;

class XrdFrmXfrQueue
{
public:

static int           Add(XrdFrmRequest *rP, XrdFrmReqFile *reqF, int theQ);

static void          Done(XrdFrmXfrJob *xP, const char *Msg);

static XrdFrmXfrJob *Get();

static int           Init();

static void          StopMon(void *parg);

                     XrdFrmXfrQueue() {}
                    ~XrdFrmXfrQueue() {}

private:

static XrdFrmXfrJob *Pull();
static int           Notify(XrdFrmRequest *rP,int qN,int rc,const char *msg=0);
static void          Send2File(char *Dest, char *Msg, int Mln);
static void          Send2UDP(char *Dest, char *Msg, int Mln);
static int           Stopped(int qNum);
static const char   *xfrName(XrdFrmRequest &reqData, int isOut);

static XrdSysMutex               hMutex;
static XrdOucHash<XrdFrmXfrJob>  hTab;

static XrdSysMutex               qMutex;
static XrdSysSemaphore           qReady;

struct theQueue
      {XrdSysSemaphore           Avail;
       XrdFrmXfrJob             *Free;
       XrdFrmXfrJob             *First;
       XrdFrmXfrJob             *Last;
              XrdSysSemaphore    Alert;
              const char        *File;
              const char        *Name;
              int                Stop;
              int                qNum;
              theQueue() : Avail(0),Free(0),First(0),Last(0),Alert(0),Stop(0) {}
             ~theQueue() {}
      };
static theQueue                  xfrQ[XrdFrmRequest::numQ];
};
#endif
