#ifndef __ODC_RESP__
#define __ODC_RESP__
/******************************************************************************/
/*                                                                            */
/*                         X r d O d c r e s p . h h                          */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                          X r d O d c R e s p C B                           */
/******************************************************************************/
  
class XrdOdcRespCB : XrdOucEICB
{
public:

void Done(int &Result, XrdOucErrInfo *eInfo) {respSync.Post();}

void Init() {while(respSync.CondWait()) {}}

int  Same(unsigned long long arg1, unsigned long long arg2) {return 0;}

void Wait() {respSync.Wait();}

     XrdOdcRespCB() : respSync(0) {}
    ~XrdOdcRespCB() {}

private:

XrdSysSemaphore     respSync;
};

/******************************************************************************/
/*                            X r d O d c R e s p                             */
/******************************************************************************/
  
class XrdOdcResp : public XrdOucErrInfo, public XrdOucEICB
{
public:
friend class XrdOdcRespQ;

static XrdOdcResp *Alloc(XrdOucErrInfo *erp, int msgid);

       void        Done(int &Result, XrdOucErrInfo *eInfo) {Recycle();}

inline int         ID() {return myID;}

       void        Reply(const char *Man, char *reply);

       int         Same(unsigned long long arg1, unsigned long long arg2)
                       {return 0;}

static void        setDelay(int repdly) {RepDelay = repdly;}

       XrdOdcResp() : XrdOucErrInfo(UserID) {next = 0;}
      ~XrdOdcResp() {}

private:
       void Recycle();

static XrdOdcResp            *nextFree;
static XrdSysMutex            myMutex;  // Protects above and below
static int                    numFree;
static const int              maxFree = 300;
static int                    RepDelay;

XrdOdcRespCB        SyncCB;
XrdOdcResp         *next;
int                 myID;
char                UserID[64];
};
  
/******************************************************************************/
/*                           X r d O d c R e s p Q                            */
/******************************************************************************/
  
class XrdOdcRespQ
{
public:
       void        Add(XrdOdcResp *rp);

       void        Purge();

       XrdOdcResp *Rem(int msgid);

       XrdOdcRespQ();
      ~XrdOdcRespQ() {Purge();}

private:

       XrdSysMutex            myMutex;  // Protects above and below
static const int              mqSize = 512;

XrdOdcResp         *mqTab[mqSize];
};
#endif
