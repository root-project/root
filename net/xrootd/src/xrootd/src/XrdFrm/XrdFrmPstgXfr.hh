#ifndef __FRMPSTGXFR_H__
#define __FRMPSTGXFR_H__
/******************************************************************************/
/*                                                                            */
/*                      X r d F r m P s t g X f r . h h                       */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"

struct Request;
struct XrdFrmPstgXrq;
class  XrdOucEnv;
class  XrdOucProg;

class XrdFrmPstgXfr
{
public:

static int  Init();

static int  Queue(XrdFrmPstgReq::Request *rP, int slot);

       void Start();

       XrdFrmPstgXfr();
      ~XrdFrmPstgXfr() {}

private:
static int  Notify(XrdFrmPstgReq::Request *rP, int rc, const char *msg=0);
static void Send2File(char *Dest, char *Msg, int Mln);
static void Send2UDP(char *Dest, char *Msg, int Mln);
const char *Stage(XrdFrmPstgXrq *xP, int &retcode);
int         StageCmd(XrdFrmPstgXrq *xP, XrdOucEnv *theEnv);
const char *StageOpt(XrdFrmPstgXrq *xP);

static XrdSysMutex               hMutex;
static XrdOucHash<XrdFrmPstgXrq> hTab;

static XrdSysMutex               qMutex;
static XrdSysSemaphore           qReady;
static XrdSysSemaphore           qAvail;
static struct XrdFrmPstgXrq     *First;
static struct XrdFrmPstgXrq     *Last;
static struct XrdFrmPstgXrq     *Free;

XrdOucProg *xfrCmd;
char cmdBuff[4096];
};
#endif
