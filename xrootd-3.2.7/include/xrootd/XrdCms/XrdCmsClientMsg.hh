#ifndef __CMS_CLIENTMSG__
#define __CMS_CLIENTMSG__
/******************************************************************************/
/*                                                                            */
/*                    X r d C m s C l i e n t M s g . h h                     */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include "XProtocol/YProtocol.hh"

#include "XrdSys/XrdSysPthread.hh"

class  XrdOucErrInfo;
class  XrdNetBuffer;

class XrdCmsClientMsg
{
public:

static XrdCmsClientMsg *Alloc(XrdOucErrInfo *erp);

inline int       getResult() {return Result;}

inline int       ID() {return id;}

static int       Init();

static int       inQ() {return numinQ;}

       void      Lock() {Hold.Lock();}

       void      Recycle();

static int       Reply(const char *Man,XrdCms::CmsRRHdr &hdr,XrdNetBuffer *buff);

       void      UnLock() {Hold.UnLock();}

       int       Wait4Reply(int wtime) {return Hold.Wait(wtime);}

      XrdCmsClientMsg() : Hold(0) {next = 0; inwaitq = 0; Resp = 0; Result = 0;}
     ~XrdCmsClientMsg() {}

private:
static const int          MidMask = 1023;
static const int          MaxMsgs = 1024;
static const int          MidIncr = 1024;
static const int          IncMask = 0x3ffffc00;
static XrdCmsClientMsg   *RemFromWaitQ(int msgid);

static int                nextid;
static int                numinQ;

static XrdCmsClientMsg   *msgTab;
static XrdCmsClientMsg   *nextfree;
static XrdSysMutex        FreeMsgQ;

XrdCmsClientMsg          *next;
XrdSysCondVar             Hold;
int                       inwaitq;
int                       id;
XrdOucErrInfo            *Resp;
int                       Result;
};
#endif
