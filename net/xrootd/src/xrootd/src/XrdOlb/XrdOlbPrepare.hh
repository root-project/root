#ifndef __OLB_PREPARE__H
#define __OLB_PREPARE__H
/******************************************************************************/
/*                                                                            */
/*                      X r d O l b P r e p a r e . h h                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$
  
#include "Xrd/XrdJob.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdOlb/XrdOlbPrepArgs.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucStream.hh"

class XrdOucMsubs;
class XrdOucName2Name;

class XrdOlbPrepare : public XrdJob
{
public:

int        Add(XrdOlbPrepArgs &pargs);

int        Del(char *reqid);

int        Exists(char *path);

void       Gone(char *path);

void       DoIt() {Scrub();
                   if (prepif) 
                      SchedP->Schedule((XrdJob *)this, scrubtime+time(0));
                  }

int        Pending() {return NumFiles;}

void       Queue(XrdOlbPrepArgs *parg);

int        Reset();

int        setParms(int rcnt, int stime, int deco=0);

int        setParms(char *ifpgm, char *ifmsg=0);

int        setParms(XrdScheduler *sp) {SchedP = sp; return 0;}

int        setParms(XrdOucName2Name *n2n) {N2N = n2n; return 0;}

           XrdOlbPrepare();
          ~XrdOlbPrepare() {}   // Never gets deleted

private:

void       Scrub();
int        startIF();

XrdSysMutex           PTMutex;
XrdOucHash<char>      PTable;
XrdOucStream          prepSched;
XrdScheduler         *SchedP;
XrdOucName2Name      *N2N;
XrdOucMsubs          *prepMsg;
time_t                lastemsg;
pid_t                 preppid;
int                   NumFiles;
int                   doEcho;
int                   resetcnt;
int                   scrub2rst;
int                   scrubtime;
char                 *prepif;
};

namespace XrdOlb
{
extern    XrdOlbPrepare PrepQ;
}
#endif
