#ifndef __OLB_CACHE__H
#define __OLB_CACHE__H
/******************************************************************************/
/*                                                                            */
/*                        X r d O l b C a c h e . h h                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$
  
#include "Xrd/XrdJob.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdOlb/XrdOlbPList.hh"
#include "XrdOlb/XrdOlbTypes.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"
 
/******************************************************************************/
/*                     S t r u c t   o o l b _ C I n f o                      */
/******************************************************************************/
  
struct XrdOlbCInfo
       {SMask_t hfvec;    // Servers who are staging or have the file
        SMask_t pfvec;    // Servers who are staging         the file
        SMask_t sbvec;    // Servers that are suspect (eventually TOE clock)
        int     deadline;
        short   roPend;   // Redirectors waiting for R/O response
        short   rwPend;   // Redirectors waiting for R/W response

        XrdOlbCInfo() {roPend = rwPend = 0;}
       ~XrdOlbCInfo();
       };

/******************************************************************************/
/*                      C l a s s   o o l b _ C a c h e                       */
/******************************************************************************/

class XrdOlbRRQInfo;
  
class XrdOlbCache
{
public:
friend class XrdOlbCache_Scrubber;

XrdOlbPList_Anchor Paths;

// AddFile() returns true if this is the first addition, false otherwise
//
int        AddFile(const char *path, SMask_t mask, int isrw=-1, 
                   int dltime=0, XrdOlbRRQInfo *Info=0);

// DelCache() deletes a specific cache line
//
void       DelCache(const char *path);

// DelFile() returns true if this is the last deletion, false otherwise
//
int        DelFile(const char *path, SMask_t mask, int dltime=0);

// GetFile() returns true if we actually found the file
//
int        GetFile(const char *path, XrdOlbCInfo &cinfo,
                   int isrw=0, XrdOlbRRQInfo *Info=0);

void       Apply(int (*func)(const char *, XrdOlbCInfo *, void *), void *Arg);

void       Bounce(SMask_t mask, char *path=0);

void       Extract(const char *pathpfx, XrdOucHash<char> *hashp);

void       Reset(int servid);

void       Scrub();

void       setLifetime(int lsec) {LifeTime = lsec;}

           XrdOlbCache() {LifeTime = 8*60*60;}
          ~XrdOlbCache() {}   // Never gets deleted

private:

void                    Add2Q(XrdOlbRRQInfo *Info, XrdOlbCInfo *cp, int isrw);
void                    Dispatch(XrdOlbCInfo *cinfo, short roQ, short rwQ);
XrdSysMutex             PTMutex;
XrdOucHash<XrdOlbCInfo> PTable;
int                     LifeTime;
};
 
/******************************************************************************/
/*             C l a s s   o o l b _ C a c h e _ S c r u b b e r              */
/******************************************************************************/
  
class XrdOlbCache_Scrubber : public XrdJob
{
public:

void  DoIt() {CacheP->Scrub();
              SchedP->Schedule((XrdJob *)this, CacheP->LifeTime+time(0));
             }
      XrdOlbCache_Scrubber(XrdOlbCache *cp, XrdScheduler *sp)
                        : XrdJob("File cache scrubber")
                {CacheP = cp; SchedP = sp;}
     ~XrdOlbCache_Scrubber() {}

private:

XrdScheduler    *SchedP;
XrdOlbCache     *CacheP;
};

namespace XrdOlb
{
extern    XrdOlbCache Cache;
}
#endif
