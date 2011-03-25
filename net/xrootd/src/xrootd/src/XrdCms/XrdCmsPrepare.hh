#ifndef __CMS_PREPARE__H
#define __CMS_PREPARE__H
/******************************************************************************/
/*                                                                            */
/*                      X r d C m s P r e p a r e . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
#include "Xrd/XrdJob.hh"
#include "Xrd/XrdScheduler.hh"

#include "XrdCms/XrdCmsPrepArgs.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdFrmProxy;
class XrdNetMsg;
class XrdOucMsubs;
class XrdOucName2Name;

class XrdCmsPrepare : public XrdJob
{
public:

int        Add(XrdCmsPrepArgs &pargs);

int        Del(char *reqid);

int        Exists(char *path);

void       Gone(char *path);

void       DoIt();

void       Inform(const char *cmd, XrdCmsPrepArgs *pargs);

int        isOK() {return prepOK;}

int        Pending() {return NumFiles;}

void       Prepare(XrdCmsPrepArgs *pargs);

void       Reset(const char *iName, const char *aPath, int aMode);

int        setParms(int rcnt, int stime, int deco=0);

int        setParms(const char *ifpgm, char *ifmsg=0);

int        setParms(XrdOucName2Name *n2n) {N2N = n2n; return 0;}

           XrdCmsPrepare();
          ~XrdCmsPrepare() {}   // Never gets deleted

private:

int        isOnline(char *path);
void       Reset();
void       Scrub();
int        startIF();

XrdSysMutex           PTMutex;
XrdOucHash<char>      PTable;
XrdOucStream          prepSched;
XrdOucName2Name      *N2N;
XrdOucMsubs          *prepMsg;
XrdNetMsg            *Relay;
XrdFrmProxy          *PrepFrm;
char                 *prepif;
time_t                lastemsg;
pid_t                 preppid;
int                   prepOK;
int                   NumFiles;
int                   doEcho;
int                   resetcnt;
int                   scrub2rst;
int                   scrubtime;
};

namespace XrdCms
{
extern    XrdCmsPrepare PrepQ;
}
#endif
