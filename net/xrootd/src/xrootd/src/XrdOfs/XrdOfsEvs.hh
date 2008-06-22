#ifndef __XRDOFSEVS_H__
#define __XRDOFSEVS_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d O f s E v s . h h                           */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/*             Based on code developed by Derek Feichtinger, CERN.            */
/******************************************************************************/
  
//         $Id$

#include <strings.h>
#include "XrdSys/XrdSysPthread.hh"

class XrdOfsEvsMsg;
class XrdOucEnv;
class XrdOucProg;
class XrdSysError;

/******************************************************************************/
/*                         X r d O f s E v s I n f o                          */
/******************************************************************************/
  
class XrdOfsEvsInfo
{
public:

enum evArg {evTID=0, evLFN1, evCGI1, evLFN2, evCGI2, evFMODE, evFSIZE, evARGS};

inline long long   FSize() {return theFSize;}

inline mode_t      FMode() {return theFMode;}

inline void        Set(evArg aNum, const char *aVal) {Arg[aNum] = aVal;}

inline const char *Val(evArg aNum) {return Arg[aNum];}

 XrdOfsEvsInfo(const char *tid,
               const char *lfn1,         const char *cgi1="", XrdOucEnv *env1=0,
               mode_t      mode=0,       long long   fsize=0,
               const char *lfn2="$LFN2", const char *cgi2="", XrdOucEnv *env2=0)
              {Arg[evTID]  = tid;
               Arg[evLFN1] = lfn1;
               Arg[evCGI1] = (cgi1 ? cgi1  : ""); Env1 = env1;
               Arg[evLFN2] = (lfn2 ? lfn2 : "$LFN2");
               Arg[evCGI2] = (cgi2  ? cgi2  : ""); Env2 = env2;
               theFMode = mode; theFSize = fsize;
              }

~XrdOfsEvsInfo() {}

private:

const char *Arg[evARGS];
XrdOucEnv  *Env1;
XrdOucEnv  *Env2;
long long   theFSize;
mode_t      theFMode;
};

/******************************************************************************/
/*                       X r d O f s E v s F o r m a t                        */
/******************************************************************************/

class XrdOfsEvsFormat
{
public:

enum evFlags {Null = 0, freeFmt = 1, cvtMode = 2, cvtFSize = 4};

const char                *Format;
      evFlags              Flags;
      XrdOfsEvsInfo::evArg Args[XrdOfsEvsInfo::evARGS];

      int     SNP(XrdOfsEvsInfo &Info, char *buff, int blen)
                     {return snprintf(buff,blen,Format, Info.Val(Args[0]),
                                     Info.Val(Args[1]), Info.Val(Args[2]),
                                     Info.Val(Args[3]), Info.Val(Args[4]),
                                     Info.Val(Args[5]), Info.Val(Args[6]));
                     }

      void    Def(evFlags theFlags, const char *Fmt, ...);

      void    Set(evFlags theFlags, const char *Fmt, int *fullArgs)
                 {if (Format && Flags & freeFmt) free((char *)Format);
                  Format = Fmt; Flags = theFlags;
                  memcpy(Args, fullArgs, sizeof(Args));
                 }

      XrdOfsEvsFormat() : Format(0), Flags(Null) {}
     ~XrdOfsEvsFormat() {}
};

/******************************************************************************/
/*                             X r d O f s E v s                              */
/******************************************************************************/
  
class XrdOfsEvs
{
public:

enum Event {All    = 0x7fffff00, None   = 0x00000000,
            Chmod  = 0x00000100, Closer = 0x00000201,
            Closew = 0x00000402, Close  = 0x00000600,
            Create = 0x00000803, Fwrite = 0x00001004,
            Mkdir  = 0x00002005, Mv     = 0x00004006,
            Openr  = 0x00008007, Openw  = 0x00010008,
            Open   = 0x00018000, Rm     = 0x00020009,
            Rmdir  = 0x0004000a, Trunc  = 0x0008000b,
            nCount = 12,
            Mask   = 0X000000ff, enMask = 0x7fffff00
           };

static const int   minMsgSize = 1360; // (16+320+1024)
static const int   maxMsgSize = 2384; // (16+320+1024+1024);

int         Enabled(Event theEvents) {return theEvents & enEvents;}

int         maxSmsg() {return maxMin;}
int         maxLmsg() {return maxMax;}

void        Notify(Event eNum, XrdOfsEvsInfo &Info);

static int  Parse(XrdSysError &Eroute, Event eNum, char *mText);

const char *Prog() {return theTarget;}

void        sendEvents(void);

int         Start(XrdSysError *eobj);

      XrdOfsEvs(Event theEvents, const char *Target, int minq=90, int maxq=10);
     ~XrdOfsEvs();

private:
const char     *eName(int eNum);
int             Feed(const char *data, int dlen);
XrdOfsEvsMsg   *getMsg(int bigmsg);
void            retMsg(XrdOfsEvsMsg *tp);

static XrdOfsEvsFormat MsgFmt[XrdOfsEvs::nCount];

pthread_t       tid;
char           *theTarget;
Event           enEvents;
XrdSysError    *eDest;
XrdOucProg     *theProg;
XrdSysMutex     qMut;
XrdSysSemaphore qSem;
XrdOfsEvsMsg   *msgFirst;
XrdOfsEvsMsg   *msgLast;
XrdSysMutex     fMut;
XrdOfsEvsMsg   *msgFreeMax;
XrdOfsEvsMsg   *msgFreeMin;
int             endIT;
int             msgFD;
int             numMax;
int             maxMax;
int             numMin;
int             maxMin;
};
#endif
