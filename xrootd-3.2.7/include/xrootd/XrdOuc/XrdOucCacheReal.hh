#ifndef __XRDOUCCACHEREAL_HH__
#define __XRDOUCCACHEREAL_HH__
/******************************************************************************/
/*                                                                            */
/*                    X r d O u c C a c h e R e a l . h h                     */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOuc/XrdOucCacheDram.hh"
#include "XrdOuc/XrdOucCacheSlot.hh"
#include "XrdSys/XrdSysPthread.hh"

/* This class defines an actual implementation of an XrdOucCache object. */

class XrdOucCacheReal : public XrdOucCacheDram
{
friend class XrdOucCacheData;
public:

XrdOucCacheIO *Attach(XrdOucCacheIO *ioP, int Options=0);

int            isAttached() {int n;
                             CMutex.Lock(); n = Attached; CMutex.UnLock();
                             return n;
                            }

               XrdOucCacheReal(int     &rc,     // Success = 0; o/w !0
                               Parms   &Parms,  // Parameters
               XrdOucCacheIO::aprParms *aprP=0);

              ~XrdOucCacheReal();

void           PreRead();

private:

void      eMsg(const char *Path, const char *What, long long xOff,
               int xLen, int ec);
int       Detach(XrdOucCacheIO *ioP);
char     *Get(XrdOucCacheIO *ioP, long long lAddr, int &rGot, int &bIO);

int       ioAdd(XrdOucCacheIO *KeyVal, int &iNum);
int       ioDel(XrdOucCacheIO *KeyVal, int &iNum);

inline
int       ioEnt(XrdOucCacheIO *kVal)
               {union {short sV[4]; XrdOucCacheIO *pV;} Key = {{0,0,0,0}};
                Key.pV = kVal;
                return ((Key.sV[0]^Key.sV[1]^Key.sV[2]^Key.sV[3])&0x7fff)%hMax;
               }
inline
int       ioLookup(int &pip, int hip, void *kval)
                  {pip = 0;
                   while(hip && kval != Slots[hip].Key)
                        {pip = hip; hip = Slots[hip].HLink;}
                   return hip;
                  }

int       Ref(char *Addr, int rAmt, int sFlags=0);
void      Trunc(XrdOucCacheIO *ioP, long long lAddr);
void      Upd(char *Addr, int wAmt, int wOff);

static const long long Shift = 48;
static const long long Strip = 0x00000000ffffffffLL;  //
static const long long MaxFO = 0x000007ffffffffffLL;  // Min 4K page -> 8TB-1

XrdOucCacheIO::aprParms aprDefault; // Default automatic preread

XrdSysMutex      CMutex;
XrdOucCacheSlot *Slots;       // 1-to-1 slot to memory map
int             *Slash;       // Slot hash table
char            *Base;        // Base of memory cache
long long        HNum;
long long        SegCnt;
long long        SegSize;
long long        OffMask;     // SegSize - 1
long long        SegShft;     // log2(SegSize)
int              SegFull;     // SegSize to mark
int              maxCache;    // Maximum read to cache
int              maxFiles;    // Maximum number of files to support
int              Options;

// The following supports CacheIO object tracking
//
int             *hTab;        // -> Hash Table
int              hMax;        // Number of entries in table
int              sFree;       // Index of free file slot
int              sBeg;        // Index of file slot array in slot table
int              sEnd;        // Last index + 1

// Various options
//
char             Dbg;         // Debug setting
char             Lgs;         // Log statistics

// This is the attach/detach control area
//
XrdSysSemaphore *AZero;
int              Attached;

// This is the pre-read control area
//
struct prTask
      {prTask          *Next;
       XrdOucCacheData *Data;
      };
void             PreRead(XrdOucCacheReal::prTask *prReq);
prTask          *prFirst;
prTask          *prLast;
XrdSysMutex      prMutex;
XrdSysSemaphore  prReady;
XrdSysSemaphore *prStop;
int              prNum;
};
#endif
