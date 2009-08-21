#ifndef __XRD_STATS_H__
#define __XRD_STATS_H__
/******************************************************************************/
/*                                                                            */
/*                           X r d S t a t s . h h                            */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <stdlib.h>

#include "XrdSys/XrdSysPthread.hh"

#define XRD_STATS_ALL    0x000000FF
#define XRD_STATS_INFO   0x00000001
#define XRD_STATS_BUFF   0x00000002
#define XRD_STATS_LINK   0x00000004
#define XRD_STATS_POLL   0x00000008
#define XRD_STATS_PROC   0x00000010
#define XRD_STATS_PROT   0x00000020
#define XRD_STATS_SCHD   0x00000040
#define XRD_STATS_SGEN   0x00000080
#define XRD_STATS_SYNC   0x40000000
#define XRD_STATS_SYNCA  0x20000000

class XrdStats
{
public:

void  Report(char **Dest=0, int iVal=600, int Opts=0);

void  Lock() {statsMutex.Lock();}       // Call before doing Stats()

const char *Stats(int opts);

void  UnLock() {statsMutex.UnLock();}   // Call after inspecting buffer

      XrdStats(const char *hn, int port, const char *in, const char *pn);

     ~XrdStats() {if (buff) free(buff);}

private:

int        InfoStats(char *buff, int blen, int dosync=0);
int        ProcStats(char *buff, int blen, int dosync=0);

static long tBoot;       // Time at boot time

XrdSysMutex statsMutex;

char       *buff;        // Used by all callers
int         blen;
int         Hlen;
char       *Head;
const char *myHost;
const char *myName;
int         myPort;
};
#endif
