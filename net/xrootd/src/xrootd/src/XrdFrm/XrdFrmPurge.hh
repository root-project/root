#ifndef __FRMPURGE__
#define __FRMPURGE__
/******************************************************************************/
/*                                                                            */
/*                        X r d F r m P u r g e . h h                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <time.h>
#include <sys/types.h>

#include "XrdFrm/XrdFrmTSort.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOuc/XrdOucHash.hh"

class XrdFrmFileset;
class XrdOucPolProg;
class XrdOucStream;
class XrdOucTList;

class XrdFrmPurge
{
public:

static void          Display();

static int           Init(XrdOucTList *sP=0, long long minV=-1, int hVal=-1);

static XrdFrmPurge  *Policy(const char *sname) {return Find(sname);}
static XrdFrmPurge  *Policy(const char *sname, long long minV, long long maxV,
                                               int hVal, int xVal);

static void          Purge();

                     XrdFrmPurge(const char *snp, XrdFrmPurge *spp=0);
                    ~XrdFrmPurge() {Clear();}

private:

// Methods
//
static void          Add(XrdFrmFileset *fsp);
       XrdFrmFileset*Advance();
       void          Clear();
       void          Defer(XrdFrmFileset *sP, time_t xTime);
const  char         *Eligible(XrdFrmFileset *sP, time_t &xTime, int hTime=0);
static XrdFrmPurge  *Find(const char *snp);
static int           LowOnSpace();
       int           PurgeFile();
static void          Remfix(const char *Ftype, const char *Fname);
static void          Scan();
static int           Screen(XrdFrmFileset *sP, int needLF);
static void          Stats(int Final);
       void          Track(XrdFrmFileset *sP);
const  char         *XPolOK(XrdFrmFileset *sP);
static XrdOucProg   *PolProg;
static XrdOucStream *PolStream;

// Static Variables

static XrdOucHash<char> BadFiles;
static time_t        lastReset;
static time_t        nextReset;

static XrdFrmPurge  *First;
static XrdFrmPurge  *Default;

static int           Left2Do;

// Variables local to each object
//
long long            freeSpace;      // Current free space
long long            fconMaxsp;      // Current free space contiguous
long long            usedSpace;      // Curreny used space (if supported)
long long            pmaxSpace;      // PMax  space (computed once)
long long            totlSpace;      // Total space (computed once)
long long            contSpace;      // Total contg (computed once)
long long            purgBytes;      // Purged bytes on last purge cycle
long long            minFSpace;      // Minimum free space
long long            maxFSpace;      // Maximum free space (what we purge to)
char                *spaceTotl;
char                *spaceTotP;
int                  spaceTLen;
int                  spaceTLep;
int                  Hold;           // Hold value
int                  Hold2x;         // Hold x2 (what we actually use)
int                  Ext;            // External policy applies
int                  numFiles;       // Total number of files
int                  prgFiles;       // Total number of purged
int                  Enabled;
int                  Stop;
int                  SNlen;

XrdFrmPurge         *Next;
XrdFrmTSort          FSTab;
char                 SName[XrdOssSpace::minSNbsz];

static const int     DeferQsz = 16;
XrdFrmFileset       *DeferQ[DeferQsz];
time_t               DeferT[DeferQsz];
};
#endif
