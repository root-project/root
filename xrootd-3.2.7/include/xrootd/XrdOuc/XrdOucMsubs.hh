#ifndef __XRDOUCMSUBS_H__
#define __XRDOUCMSUBS_H__
/******************************************************************************/
/*                                                                            */
/*                        X r d O u c M S u b s . h h                         */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdlib.h>

#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucName2Name.hh"

/******************************************************************************/
/*      P r e d e f i n e d   E n v i r o n m e n t   V a r i a b l e s       */
/******************************************************************************/

#define CMS_CID             "cms&cid"
#define SEC_USER            "sec&user"
#define SEC_HOST            "sec&host"
#define SEC_POLICY          "sec&policy"
#define XRD_INS             "xrd&ins"
  
/******************************************************************************/
/*                           X r d O u c M s u b s                            */
/******************************************************************************/
  
struct XrdOucMsubsInfo
{
const char      *Tid;       // $TID   or $RID  unless Rid is defined.
XrdOucEnv       *Env;
XrdOucName2Name *N2N;
const char      *lfn;       // $LFN
const char      *lfn2;      // $LFN2  or $NOTIFY or $SRC
const char      *pfn;       // $PFN
const char      *pfn2;      // $PFN2             or $DST
const char      *misc;      // $OPTS  or $MDP
const char      *Rid;       // $RID for real
const char      *Src;       // $SRC
const char      *Dst;       // $DST
char            *pfnbuff;
char            *rfnbuff;
char            *pfn2buff;
char            *rfn2buff;
mode_t           Mode;      // $FMODE or $PRTY
int              Oflag;     // $OFLAG
char             mbuff[12];
char             obuff[4];

             XrdOucMsubsInfo(const char *tid, XrdOucEnv *envP, 
                             XrdOucName2Name *n2np,
                             const char *lfnP, const char *lfn2P,
                             mode_t mode=0,    int ofl=0,
                             const char *Opts=0, const char *ridP=0,
                             const char *pfnP=0, const char *pfn2P=0)
                            : Tid(tid), Env(envP), N2N(n2np),
                              lfn(lfnP), lfn2(lfn2P), pfn(pfnP), pfn2(pfn2P),
                              misc(Opts), Rid(ridP), Mode(mode), Oflag(ofl)
                              {pfnbuff = rfnbuff = pfn2buff = rfn2buff = 0;}
            ~XrdOucMsubsInfo(){if (pfnbuff ) free(pfnbuff);
                               if (rfnbuff ) free(rfnbuff);
                               if (pfn2buff) free(pfn2buff);
                               if (rfn2buff) free(rfn2buff);
                              }
};
  
class XrdOucMsubs
{
public:

static const int maxElem = 32;

int   Parse(const char *oname, char *msg);

int   Subs(XrdOucMsubsInfo &Info, char **Data, int *Dlen);

      XrdOucMsubs(XrdSysError *errp);
     ~XrdOucMsubs();

private:
char *getVal(XrdOucMsubsInfo &Info, int vNum);

enum vNum {vLFN =  1, vPFN =  2, vRFN =  3, vLFN2 =  4, vPFN2 =  5, vRFN2 =  6,
           vFM  =  7, vOFL =  8, vUSR =  9, vHST  = 10, vTID  = 11,
           vNFY = 12, vOPT = 13, vPTY = 14, vRID  = 15, vCGI  = 16,
           vMDP = 17, vSRC = 18, vDST = 19, vCID  = 20, vINS  = 21};

static const int   vMax = 22;
static const char *vName[vMax];

XrdSysError *eDest;
char        *mText;
char        *mData[maxElem+1];
int          mDlen[maxElem+1];
int          numElem;
};
#endif
