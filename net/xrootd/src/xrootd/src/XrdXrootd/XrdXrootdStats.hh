#ifndef __XROOTD_STATS_H__
#define __XROOTD_STATS_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d X r o o t d S t a t s . h h                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//      $Id$

#include "XrdSys/XrdSysPthread.hh"

class XrdSfsFileSystem;
class XrdStats;
class XrdXrootdResponse;

class XrdXrootdStats
{
public:
int              Count;        // Stats: Number of matches
int              errorCnt;     // Stats: Number of errors returned
int              redirCnt;     // Stats: Number of redirects
int              stallCnt;     // Stats: Number of stalls
int              getfCnt;      // Stats: Number of getfiles
int              putfCnt;      // Stats: Number of putfiles
int              openCnt;      // Stats: Number of opens
int              readCnt;      // Stats: Number of reads
int              prerCnt;      // Stats: Number of reads (pre)
int              writeCnt;     // Stats: Number of writes
int              syncCnt;      // Stats: Number of sync
int              miscCnt;      // Stats: Number of miscellaneous
int              AsyncNum;     // Stats: Number of async ops
int              AsyncMax;     // Stats: Number of async max
int              AsyncRej;     // Stats: Number of async rejected
int              AsyncNow;     // Stats: Number of async now (not locked)
int              Refresh;      // Stats: Number of refresh requests

XrdSysMutex      statsMutex;   // Mutex to serialize updates

void             setFS(XrdSfsFileSystem *fsp) {fsP = fsp;}

int              Stats(char *buff, int blen, int do_sync=0);

int              Stats(XrdXrootdResponse &resp, const char *opts);

                 XrdXrootdStats(XrdStats *sp);
                ~XrdXrootdStats() {}
private:

XrdSfsFileSystem *fsP;
XrdStats *xstats;
};
#endif
