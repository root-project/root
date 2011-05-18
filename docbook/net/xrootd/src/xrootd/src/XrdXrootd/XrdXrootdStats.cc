/******************************************************************************/
/*                                                                            */
/*                     X r d X r o o t d S t a t s . c c                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

const char *XrdXrootdStatsCVSID = "$Id$";
 
#include <stdio.h>
  
#include "Xrd/XrdStats.hh"
#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdXrootd/XrdXrootdResponse.hh"
#include "XrdXrootd/XrdXrootdStats.hh"
 
/******************************************************************************/
/*                           C o n s t r c u t o r                            */
/******************************************************************************/
  
XrdXrootdStats::XrdXrootdStats(XrdStats *sp)
{

xstats   = sp;

Count    = 0;     // Stats: Number of matches
errorCnt = 0;     // Stats: Number of errors returned
redirCnt = 0;     // Stats: Number of redirects
stallCnt = 0;     // Stats: Number of stalls
getfCnt  = 0;     // Stats: Number of getfiles
putfCnt  = 0;     // Stats: Number of putfiles
openCnt  = 0;     // Stats: Number of opens
readCnt  = 0;     // Stats: Number of reads
prerCnt  = 0;     // Stats: Number of reads
writeCnt = 0;     // Stats: Number of writes
syncCnt  = 0;     // Stats: Number of sync
miscCnt  = 0;     // Stats: Number of miscellaneous
AsyncNum = 0;     // Stats: Number of async ops
AsyncMax = 0;     // Stats: Number of async max
AsyncRej = 0;     // Stats: Number of async rejected
AsyncNow = 0;     // Stats: Number of async now (not locked)
Refresh  = 0;     // Stats: Number of refresh requests
}

/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/
  
int XrdXrootdStats::Stats(char *buff, int blen, int do_sync)
{
   static const char statfmt[] = "<stats id=\"xrootd\"><num>%d</num>"
   "<ops><open>%d</open><rf>%d</rf><rd>%d</rd><pr>%d</pr><wr>%d</wr>"
   "<sync>%d</sync><getf>%d</getf><putf>%d</putf><misc>%d</misc></ops>"
   "<aio><num>%d</num><max>%d</max><rej>%d</rej></aio></stats>";
   int len;

// If no buffer, caller wants the maximum size we will generate
//
   if (!buff) return sizeof(statfmt) + (16*13) + (fsP ? fsP->getStats(0,0) : 0);

// Format our statistics
//
   statsMutex.Lock();
   len = snprintf(buff, blen, statfmt, Count, openCnt, Refresh, readCnt,
                  prerCnt, writeCnt, syncCnt, getfCnt, putfCnt, miscCnt,
                  AsyncNum, AsyncMax, AsyncRej);
   statsMutex.UnLock();

// Now include filesystem statistics and return
//
   if (fsP) len += fsP->getStats(buff+len, blen-len);
   return len;
}
 
/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/
  
int XrdXrootdStats::Stats(XrdXrootdResponse &resp, const char *opts)
{
    int i, xopts = 0;

    while(*opts)
         {switch(*opts)
                {case 'a': xopts |= XRD_STATS_ALL;  break;
                 case 'b': xopts |= XRD_STATS_BUFF; break;    // b_uff
                 case 'i': xopts |= XRD_STATS_INFO; break;    // i_nfo
                 case 'l': xopts |= XRD_STATS_LINK; break;    // l_ink
                 case 'd': xopts |= XRD_STATS_POLL; break;    // d_evice
                 case 'u': xopts |= XRD_STATS_PROC; break;    // u_sage
                 case 'p': xopts |= XRD_STATS_PROT; break;    // p_rotocol
                 case 's': xopts |= XRD_STATS_SCHD; break;    // s_scheduler
                 default:  break;
                }
          opts++;
         }

    if (!xopts) return resp.Send();

    xstats->Lock();
    i = resp.Send(xstats->Stats(xopts));
    xstats->UnLock();
    return i;
}
