#ifndef __XRDOFS_STATS_H__
#define __XRDOFS_STATS_H__
/******************************************************************************/
/*                                                                            */
/*                        X r d O f s S t a t s . h h                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include <stdlib.h>

#include "XrdSys/XrdSysPthread.hh"

class XrdOfsStats
{
public:

struct      StatsData
{
int         numOpenR;   // Read
int         numOpenW;   // Write
int         numOpenP;   // Posc
int         numUnpsist; // Posc
int         numHandles;
int         numRedirect;
int         numStarted;
int         numReplies;
int         numErrors;
int         numDelays;
int         numSeventOK;
int         numSeventER;
}           Data;

XrdSysMutex sdMutex;

inline void Add(int &Cntr) {sdMutex.Lock(); Cntr++; sdMutex.UnLock();}

inline void Dec(int &Cntr) {sdMutex.Lock(); Cntr--; sdMutex.UnLock();}

       int  Report(char *Buff, int Blen);

       void setRole(const char *theRole) {myRole = theRole;}

            XrdOfsStats() {memset(&Data, 0, sizeof(Data));}
           ~XrdOfsStats() {}

private:

const char *myRole;
};
#endif
