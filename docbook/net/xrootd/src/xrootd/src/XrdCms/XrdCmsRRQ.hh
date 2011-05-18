#ifndef __XRDCMSRRQ_HH__
#define __XRDCMSRRQ_HH__
/******************************************************************************/
/*                                                                            */
/*                          X r d C m s R R Q . h h                           */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/uio.h>

#include "XProtocol/XPtypes.hh"
#include "XProtocol/YProtocol.hh"

#include "XrdCms/XrdCmsTypes.hh"
#include "XrdOuc/XrdOucDLlist.hh"
#include "XrdSys/XrdSysPthread.hh"
  
/******************************************************************************/
/*                         X r d C m s R R Q I n f o                          */
/******************************************************************************/
  
class XrdCmsRRQInfo
{
public:
void     *Key;     // Key link, which is the cache line address
kXR_unt32 ID;      // Response link, which is the request ID
int       Rinst;   // Redirector instance
short     Rnum;    // Redirector number (RTable slot number)
char      isRW;    // True if r/w access wanted
char      isLU;    // True if locate response wanted
SMask_t   rwVec;   // R/W servers for corresponding path (if isLU is true)

        XrdCmsRRQInfo() {isLU = 0;}
        XrdCmsRRQInfo(int rinst, short rnum, kXR_unt32 id)
                        {Key = 0; ID = id; 
                         Rinst = rinst; Rnum = rnum; isRW = isLU = 0;
                        }
       ~XrdCmsRRQInfo() {}
};

/******************************************************************************/
/*                         X r d C m s R R Q S l o t                          */
/******************************************************************************/
  
class XrdCmsRRQSlot
{
friend class XrdCmsRRQ;

static XrdCmsRRQSlot *Alloc(XrdCmsRRQInfo *Info);

       void           Recycle();

       XrdCmsRRQSlot();
      ~XrdCmsRRQSlot() {}

private:

static   XrdSysMutex                 myMutex;
static   XrdCmsRRQSlot              *freeSlot;
static   short                       initSlot;

         XrdOucDLlist<XrdCmsRRQSlot> Link;
         XrdCmsRRQSlot              *Cont;
         XrdCmsRRQSlot              *LkUp;
         XrdCmsRRQInfo               Info;
         SMask_t                     Arg1;
         SMask_t                     Arg2;
unsigned int                         Expire;
         int                         slotNum;
};

/******************************************************************************/
/*                             X r d C m s R R Q                              */
/******************************************************************************/
  
class XrdCmsRRQ
{
public:

short Add(short Snum, XrdCmsRRQInfo *ip);

void  Del(short Snum, const void *Key);

int   Init(int Tint=0, int Tdly=0);

void  Ready(int Snum, const void *Key, SMask_t mask1, SMask_t mask2);

void *Respond();

void *TimeOut();

      XrdCmsRRQ() : isWaiting(0), isReady(0), Tslice(178),
                    Tdelay(5),    myClock(0) {}
     ~XrdCmsRRQ() {}

private:

void sendLocResp(XrdCmsRRQSlot *lP);
void sendResponse(XrdCmsRRQInfo *Info, int doredir, int totlen = 0);
static const int numSlots = 1024;

         XrdSysMutex                   myMutex;
         XrdSysSemaphore               isWaiting;
         XrdSysSemaphore               isReady;
         XrdCmsRRQSlot                 Slot[numSlots];
         XrdOucDLlist<XrdCmsRRQSlot>   waitQ;
         XrdOucDLlist<XrdCmsRRQSlot>   readyQ;  // Redirect/Locate ready queue
static   const int                     iov_cnt = 2;
         struct iovec                  data_iov[iov_cnt];
         struct iovec                  redr_iov[iov_cnt];
         XrdCms::CmsResponse           dataResp;
         XrdCms::CmsResponse           redrResp;
         XrdCms::CmsResponse           waitResp;
union   {char                          hostbuff[288];
         char                          databuff[XrdCms::CmsLocateRequest::RILen
                                               *STMax];
        };
         int                           Tslice;
         int                           Tdelay;
unsigned int                           myClock;
};

namespace XrdCms
{
extern    XrdCmsRRQ RRQ;
}
#endif
