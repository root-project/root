#ifndef __CMS_RESP__
#define __CMS_RESP__
/******************************************************************************/
/*                                                                            */
/*                         X r d C m s r e s p . h h                          */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

// Based on: XrdCmsResp.hh,v 1.1 2006/09/26 07:49:15 abh

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysPthread.hh"

#include "XProtocol/YProtocol.hh"

/******************************************************************************/
/*                          X r d C m s R e s p C B                           */
/******************************************************************************/

class XrdCmsRespCB : XrdOucEICB
{
public:

void Done(int &Result, XrdOucErrInfo *eInfo) {respSync.Post();}

void Init() {while(respSync.CondWait()) {}}

int  Same(unsigned long long arg1, unsigned long long arg2) {return 0;}

void Wait() {respSync.Wait();}

     XrdCmsRespCB() : respSync(0) {}
    ~XrdCmsRespCB() {}

private:

XrdSysSemaphore     respSync;
};

/******************************************************************************/
/*                            X r d C m s R e s p                             */
/******************************************************************************/

class XrdNetBuffer;
  
class XrdCmsResp : public XrdOucEICB, public XrdOucErrInfo
{
public:
friend class XrdCmsRespQ;

static XrdCmsResp *Alloc(XrdOucErrInfo *erp, int msgid);

       void        Done(int &Result, XrdOucErrInfo *eInfo) {Recycle();}

inline int         ID() {return myID;}

       void        Reply(const char   *Man, XrdCms::CmsRRHdr &rrhdr,
                         XrdNetBuffer *netbuff);

static void        Reply();

       int         Same(unsigned long long arg1, unsigned long long arg2)
                       {return 0;}

static void        setDelay(int repdly) {RepDelay = repdly;}

       XrdCmsResp() : XrdOucErrInfo(UserID) {next = 0; myBuff = 0;}
      ~XrdCmsResp() {}

private:
       void Recycle();
       void ReplyXeq();

static XrdSysSemaphore        isReady;
static XrdSysMutex            rdyMutex;  // Protects the below
static XrdCmsResp            *First;
static XrdCmsResp            *Last;

static XrdSysMutex            myMutex;  // Protects above and below
static XrdCmsResp            *nextFree;
static int                    numFree;
static const int              maxFree = 300;
static int                    RepDelay;

XrdCms::CmsRRHdr    myRRHdr;
XrdNetBuffer       *myBuff;
char                theMan[64];

XrdCmsRespCB        SyncCB;
XrdCmsResp         *next;
int                 myID;
char                UserID[64];
};
  
/******************************************************************************/
/*                           X r d O d c R e s p Q                            */
/******************************************************************************/
  
class XrdCmsRespQ
{
public:
       void        Add(XrdCmsResp *rp);

       void        Purge();

       XrdCmsResp *Rem(int msgid);

       XrdCmsRespQ();
      ~XrdCmsRespQ() {Purge();}

private:

       XrdSysMutex  myMutex;  // Protects above and below
static const int    mqSize = 512;

XrdCmsResp         *mqTab[mqSize];
};
#endif
