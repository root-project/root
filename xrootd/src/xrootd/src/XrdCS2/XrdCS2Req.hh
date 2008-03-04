#ifndef __XRDCS2REQ_H_
#define __XRDCS2REQ_H_
/******************************************************************************/
/*                                                                            */
/*                          X r d C S 2 R e q . h h                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdOlb/XrdOlbReq.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdSys/XrdSysPthread.hh"

#include "osdep.h"

struct XrdOlbXmiEnv;
class  XrdOucTrace;

class XrdCS2Req
{
public:

static XrdCS2Req  *Alloc(XrdOlbReq *ReqP, const char *Path, int as_W=0);

       void        Lock() {myMutex.Lock(); myLock = 1;}

       const char *Path() {return thePath;}

       void        Queue();

static XrdCS2Req  *Remove(const char *Path);

       XrdCS2Req  *Recycle();

       XrdOlbReq  *Request() {return olbReq;}

static void        Set(XrdOlbXmiEnv *Env);

       void        UnLock() {myLock = 0; myMutex.UnLock();}

static int         Wait4Q_R();
static int         Wait4Q_W();

                   XrdCS2Req() {}
                  ~XrdCS2Req() {}

private:

static unsigned int SlotNum(const char *Path);

static XrdSysMutex      myMutex;
static XrdSysSemaphore  mySem_R;
static XrdSysSemaphore  mySem_W;
static XrdCS2Req       *nextFree;
static const int        Slots = 64;
static XrdCS2Req       *STab[Slots];
static int              numFree;
static int              numinQ_R;
static int              numinQ_W;
static const int        maxFree   = 100;
static const int        retryTime = 60;
static XrdSysError     *eDest;         // -> Error message handler
static XrdOucTrace     *Trace;         // -> Trace handler
static XrdOucName2Name *N2N;           // -> lfn mapper

       XrdCS2Req       *Next;
       XrdCS2Req       *Same;
       XrdOlbReq       *olbReq;
       int              myLock;
       int              is_W;
static const unsigned int PathSize = 1024;
static char             isWaiting_R;
static char             isWaiting_W;
       char             thePath[PathSize];
};
#endif
