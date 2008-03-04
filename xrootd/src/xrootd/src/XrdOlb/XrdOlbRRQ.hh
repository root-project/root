#ifndef __XRDOLBRRQ_HH__
#define __XRDOLBRRQ_HH__
/******************************************************************************/
/*                                                                            */
/*                          X r d O l b R R Q . h h                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/uio.h>

#include "XrdOlb/XrdOlbTypes.hh"
#include "XrdOuc/XrdOucDLlist.hh"
#include "XrdSys/XrdSysPthread.hh"
  
/******************************************************************************/
/*                         X r d O l b R R Q I n f o                          */
/******************************************************************************/
  
class XrdOlbRRQInfo
{
public:
void   *Key;     // Key link, which is the cache line address
int     Rinst;   // Redirector instance
short   Rnum;    // Redirector number (RTable slot number)
char    isRW;    // True if r/w access wanted
char    isLU;    // True if this is a lookup only
char    ID[16];  // Response link, which is the request ID
SMask_t Arg;     // Argument to lookup

        XrdOlbRRQInfo() {isLU = 0;}
       ~XrdOlbRRQInfo() {}
};

/******************************************************************************/
/*                         X r d O l b R R Q S l o t                          */
/******************************************************************************/
  
class XrdOlbRRQSlot
{
friend class XrdOlbRRQ;

static XrdOlbRRQSlot *Alloc(XrdOlbRRQInfo *Info);

       void           Recycle();

       XrdOlbRRQSlot();
      ~XrdOlbRRQSlot() {}

private:

static   XrdSysMutex                 myMutex;
static   XrdOlbRRQSlot              *freeSlot;
static   short                       initSlot;

         XrdOucDLlist<XrdOlbRRQSlot> Link;
         XrdOlbRRQSlot              *Cont;
         XrdOlbRRQInfo               Info;
         SMask_t                     Arg;
unsigned int                         Expire;
         int                         slotNum;
};

/******************************************************************************/
/*                             X r d O l b R R Q                              */
/******************************************************************************/
  
class XrdOlbRRQ
{
public:

short Add(short Snum, XrdOlbRRQInfo *ip);

void  Del(short Snum, const void *Key);

int   Init(int Tint=0, int Tdly=0);

void  Ready(int Snum, const void *Key, SMask_t mask);

void *Respond();

void *TimeOut();

      XrdOlbRRQ() : isWaiting(0), isReady(0), Tslice(133),
                    Tdelay(5),    myClock(0) {}
     ~XrdOlbRRQ() {}

private:

XrdOlbRRQSlot *sendLocInfo(XrdOlbRRQSlot *Sp);
void           sendResponse(XrdOlbRRQInfo *Info, int doredir);
static const int numSlots = 1024;

         XrdSysMutex                 myMutex;
         XrdSysSemaphore             isWaiting;
         XrdSysSemaphore             isReady;
         XrdOlbRRQSlot               Slot[numSlots];
         XrdOucDLlist<XrdOlbRRQSlot> waitQ;
         XrdOucDLlist<XrdOlbRRQSlot> readyQ;
static   const int                   redr_iov_cnt = 3;
static   const int                   wait_iov_cnt = 2;
         struct iovec                redr_iov[redr_iov_cnt];
         struct iovec                wait_iov[wait_iov_cnt];
         char                        hostbuff[288];
         char                        waitbuff[16];
         int                         Tslice;
         int                         Tdelay;
unsigned int                         myClock;
};

namespace XrdOlb
{
extern    XrdOlbRRQ RRQ;
}
#endif
