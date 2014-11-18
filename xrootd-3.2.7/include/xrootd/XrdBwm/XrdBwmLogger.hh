#ifndef __XRDBWMLOGGER_H__
#define __XRDBWMLOGGER_H__
/******************************************************************************/
/*                                                                            */
/*                       X r d B w m L o g g e r . h h                        */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <strings.h>
#include "XrdSys/XrdSysPthread.hh"

class XrdBwmLoggerMsg;
class XrdOucProg;
class XrdSysError;
  
class XrdBwmLogger
{
public:

struct Info
      {const char   *Tident;
       const char   *Lfn;
       const char   *lclNode;
       const char   *rmtNode;
             time_t  ATime;    // Arrival
             time_t  BTime;    // Begin
             time_t  CTime;    // Complete
             int     numqIn;
             int     numqOut;
             int     numqXeq;
        long long    Size;
             int     ESec;
             char    Flow;     // I or O
      };

void        Event(Info &eInfo);

const char *Prog() {return theTarget;}

void        sendEvents(void);

int         Start(XrdSysError *eobj);

      XrdBwmLogger(const char *Target);
     ~XrdBwmLogger();

private:
int              Feed(const char *data, int dlen);
XrdBwmLoggerMsg *getMsg();
void             retMsg(XrdBwmLoggerMsg *tp);

pthread_t          tid;
char              *theTarget;
XrdSysError       *eDest;
XrdOucProg        *theProg;
XrdSysMutex        qMut;
XrdSysSemaphore    qSem;
XrdBwmLoggerMsg   *msgFirst;
XrdBwmLoggerMsg   *msgLast;
XrdSysMutex        fMut;
XrdBwmLoggerMsg   *msgFree;
int                msgFD;
int                endIT;
int                msgsInQ;
static const int   maxmInQ = 256;
char               theEOL;
};
#endif
