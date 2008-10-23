#ifndef __XRDCnsEvent_H_
#define __XRDCnsEvent_H_
/******************************************************************************/
/*                                                                            */
/*                        X r d C n s E v e n t . h h                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdCnsEvent
{
public:

static const char evClosew = 't';
static const char evCreate = 'c';
static const char evMkdir  = 'd';
static const char evMv     = 'm';
static const char evRm     = 'r';
static const char evRmdir  = 'D';

static const int  lfnBSize = 2050; // (2 * 1024 + 2)

static XrdCnsEvent *Alloc();

static int          Init(const char *aP, const char *pP, int qLim);

inline const char  *Lfn1() {return Event.lfnBuff;}

inline const char  *Lfn2() {return Event.lfnBuff+Event.lfn1Len;}

inline int          Mode() {return Event.Mode;}

       void         Queue();

       void         Recycle();

static XrdCnsEvent *Remove(unsigned char &eT);

       void         set(const char *pP) 
                       {pfxLen = strlcpy(pfxPath, pP, sizeof(pfxPath));
                        maxLen1 = lfnBSize - pfxLen;
                       }

inline void         setMode(mode_t    Mode) {Event.Mode = Mode;}

inline void         setSize(long long Size) {Event.Size = Size;}

inline void         setType(const char evt) {Event.Type = evt;}

       int          setType(const char *evt);

inline int          setLfn1(const char *lfn)
                           {int n;
                            strcpy(Event.lfnBuff, pfxPath);
                            n = strlcpy(Event.lfnBuff+pfxLen, lfn, maxLen1);
                            if (n >= maxLen1) return 0;
                            Event.lfn1Len  += pfxLen+n+1;
                            EventLen  = HdrSize + Event.lfn1Len;
                            return 1;
                           }

inline int          setLfn2(const char *lfn)
                           {int n, k = lfnBSize - (Event.lfn1Len + pfxLen);
                            strcpy(Event.lfnBuff+Event.lfn1Len, pfxPath);
                            n=strlcpy(Event.lfnBuff+Event.lfn1Len+pfxLen,lfn,k);
                            if (n >= k) return 0;
                            EventLen += pfxLen+n+1;
                            return 1;
                           }

inline long long    Size() {return Event.Size;}

       const char   Type() {return Event.Type;}

                    XrdCnsEvent() {strcpy(Event.lfnBuff, pfxPath);}
                   ~XrdCnsEvent() {}

private:

static int Recover(const char *, off_t);

XrdCnsEvent  *Next;


struct EventRec 
       {unsigned int     Number;
        short            lfn1Len;
        char             Type;
        char             Pad;
        union {long long Size;
               int       Mode;
              };
        char             lfnBuff[lfnBSize];
       };

static const int     HdrSize = sizeof(EventRec) - lfnBSize;

EventRec             Event;
int                  EventLen;
int                  EventOff;

static unsigned int  EventNumber;

static int    maxLen1;  // lfnBSize - pfxLen
static int    pfxLen;
static char   pfxPath[1024];

static XrdSysMutex      aqMutex;
static XrdSysMutex      dqMutex;
static XrdSysSemaphore  mySem;
static XrdCnsEvent     *freeEvent;
static XrdCnsEvent     *frstEvent;
static XrdCnsEvent     *lastEvent;

static char             logFN[1024];
static int              logFD;
static int              logOffset;
static int              logOffmax;
static char             Running;
};
#endif
