#ifndef __ODC_MANAGER__
#define __ODC_MANAGER__
/******************************************************************************/
/*                                                                            */
/*                      X r d O d c M a n a g e r . h h                       */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

#include <stdio.h>
#include <sys/uio.h>

#include "XrdOdc/XrdOdcResp.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdSysError;
class XrdSysLogger;
class XrdNetLink;
class XrdNetWork;

class XrdOdcManager
{
public:

int            delayResp(XrdOucErrInfo &Resp);

void           relayResp(int msgid, char *msg);

int            isActive() {return Active;}

XrdOdcManager *nextManager() {return Next;}

char          *Name() {return Host;}
char          *NPfx() {return HPfx;}

int            Send(char *msg, int mlen=0);
int            Send(const struct iovec *iov, int iovcnt);

void           setTID(pthread_t tid) {mytid = tid;}

void          *Start();

void           setNext(XrdOdcManager *np) {Next = np;}

void           whatsUp();

               XrdOdcManager(XrdSysError *erp, char *host, int port, 
                             int cw, int nr);
              ~XrdOdcManager();

private:
void  Hookup();
void  Sleep(int slpsec);
char *Receive(int &msgid);

XrdSysSemaphore syncResp;
XrdOdcRespQ     RespQ;

XrdOdcManager *Next;
XrdSysMutex    myData;
XrdSysError   *eDest;
XrdNetLink    *Link;
XrdNetWork    *Network;
char          *Host;
char          *HPfx;
int            Port;
pthread_t      mytid;
int            dally;
int            Active;
int            Silent;
int            nrMax;
int            maxMsgID;
};
#endif
