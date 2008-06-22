#ifndef __XRDOFSEVR_H__
#define __XRDOFSEVR_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d O f s E v r . h h                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <strings.h>
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucStream.hh"

class XrdSysError;
class XrdCmsClient;

class XrdOfsEvr
{
public:
class theClient;

void flushEvents();

int  Init(XrdSysError *eObj, XrdCmsClient *trg=0);

void recvEvents();

void Wait4Event(const char *path, XrdOucErrInfo *einfo);

void Work4Event(theClient *Client);

      XrdOfsEvr() : mySem(0) {runQ = 0; deferQ = 0;}
     ~XrdOfsEvr();

class theClient : XrdOucEICB
{
public:

void Done(int &Result, XrdOucErrInfo *eInfo) {EvrP->Work4Event(this);}

int  Same(unsigned long long arg1, unsigned long long arg2) {return 0;}

theClient         *Next;
const char        *User;
char              *Path;
XrdOfsEvr         *EvrP;
XrdOucEICB        *evtCB;
unsigned long long evtCBarg;

         theClient(XrdOfsEvr *evr, XrdOucErrInfo *einfo, const char *path=0)
                           {evtCB = einfo->getErrCB(evtCBarg);
                            User  = einfo->getErrUser();
                            Path  = (path ? strdup(path) : 0);
                            EvrP  = evr;
                            Next  = 0;
                           }
        ~theClient() {if (Path) free(Path);}
        };

struct theEvent  {theClient  *aClient;
                  char       *finalMsg;
                  int         finalRC;
                  char        Happened;

                  theEvent(int rc, const char *emsg, theClient *cp=0)
                          {aClient = cp; finalRC = rc; Happened = 0;
                           finalMsg = (emsg ? strdup(emsg) : 0);
                          }
                 ~theEvent() 
                          {if (finalMsg) free(finalMsg);
                           if (aClient) delete aClient;
                          }
                  };

private:

void eventStage();
void sendEvent(theEvent *ep);

static const int     maxLife = (8*60*60); // Eight hours
XrdSysMutex          myMutex;
XrdSysSemaphore      mySem;
XrdOucStream         eventFIFO;
XrdSysError         *eDest;
XrdCmsClient        *Balancer;
theClient           *deferQ;
int                  runQ;
int                  msgFD;

XrdOucHash<theEvent> Events;
};
#endif
