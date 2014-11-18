#ifndef __XRDOUCCACHEDATA_HH__
#define __XRDOUCCACHEDATA_HH__
/******************************************************************************/
/*                                                                            */
/*                    X r d O u c C a c h e D a t a . h h                     */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* The XrdOucCacheData object defines a remanufactured XrdOucCacheIO object and
   is used to front a XrdOucCacheIO object with an XrdOucCacheReal object.
*/

#include "XrdOuc/XrdOucCache.hh"
#include "XrdOuc/XrdOucCacheReal.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysXSLock.hh"
  
/******************************************************************************/
/*                 C l a s s   X r d O u c C a c h e D a t a                  */
/******************************************************************************/

class XrdOucCacheData : public XrdOucCacheIO
{
public:

XrdOucCacheIO *Base() {return ioObj;}

XrdOucCacheIO *Detach();

long long      FSize() {return (ioObj ? ioObj->FSize() : 0);}

const char    *Path() {return ioObj->Path();}

void           Preread();

void           Preread(aprParms &Parms);

void           Preread(long long Offs, int rLen, int Opts=0);

int            Read (char  *Buffer, long long  Offset, int  Length);

static int     setAPR(aprParms &Dest, aprParms &Src, int pSize);

int            Sync() {return 0;} // We only support write-through for now

int            Trunc(long long Offset);

int            Write(char  *Buffer, long long  Offset,  int  Length);

               XrdOucCacheData(XrdOucCacheReal *cP, XrdOucCacheIO *ioP,
                               long long    vn,     int            opts);

private:
              ~XrdOucCacheData() {}
void           QueuePR(long long SegOffs, int rLen, int prHow, int isAuto=0);
int            Read (XrdOucCacheStats &Now,
                      char *Buffer, long long Offs, int Length);

// The following is for read/write support
//
class MrSw
{
public:
inline void UnLock() {if (myLock) {myLock->UnLock(myUsage); myLock = 0;}}

            MrSw(XrdSysXSLock *lP, XrdSysXS_Type usage) : myUsage(usage)
                {if ((myLock = lP)) lP->Lock(usage);}
           ~MrSw() {if (myLock) myLock->UnLock(myUsage);}

private:
XrdSysXSLock *myLock;
XrdSysXS_Type myUsage;
};

// The following supports MRSW serialization
//
XrdSysXSLock     rwLock;
XrdSysXSLock    *pPLock;  // 0 if no preread lock required
XrdSysXSLock    *rPLock;  // 0 if no    read lock required
XrdSysXSLock    *wPLock;  // 0 if no   write lock required
XrdSysXS_Type    pPLopt;
XrdSysXS_Type    rPLopt;

XrdSysMutex      DMutex;
XrdOucCacheReal *Cache;
XrdOucCacheIO   *ioObj;
long long        VNum;
long long        SegSize;
long long        OffMask;
long long        SegShft;
int              maxCache;
char             isFIS;
char             isRW;
char             isADB;
char             Debug;

static const int okRW   = 1;
static const int xqRW   = 2;

// Preread Control Area
//
XrdOucCacheReal::prTask prReq;
XrdSysSemaphore *prStop;

long long        prNSS;          // Next Sequential Segment for maxi prereads

static const int prRRMax= 5;
long long        prRR[prRRMax];  // Recent reads
int              prRRNow;        // Pointer to next entry to use

static const int prMax  = 8;
static const int prRun  = 1;     // Status in prActive (running)
static const int prWait = 2;     // Status in prActive (waiting)

static const int prLRU  = 1;     // Status in prOpt    (set LRU)
static const int prSUSE = 2;     // Status in prOpt    (set Single Use)
static const int prSKIP = 3;     // Status in prOpt    (skip entry)

aprParms         Apr;
long long        prCalc;
long long        prBeg[prMax];
long long        prEnd[prMax];
int              prNext;
int              prFree;
int              prPerf;
char             prOpt[prMax];
char             prOK;
char             prActive;
char             prAuto;
};
#endif
