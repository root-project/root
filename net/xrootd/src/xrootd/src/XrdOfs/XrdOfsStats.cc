/******************************************************************************/
/*                                                                            */
/*                        X r d O f s S t a t s . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdOfsStatsCVSID = "$Id$";

#include <stdio.h>

#include "XrdOfs/XrdOfsStats.hh"

/******************************************************************************/
/*                                R e p o r t                                 */
/******************************************************************************/
  
int XrdOfsStats::Report(char *buff, int blen)
{
    static const char stats1[] = "<stats id=\"ofs\"><role>%s</role>"
           "<opr>%d</opr><opw>%d</opw><opp>%d</opp><ups>%d</ups><han>%d</han>"
           "<rdr>%d</rdr><bxq>%d</bxq><rep>%d</rep><err>%d</err><dly>%d</dly>"
           "<sok>%d</sok><ser>%d</ser></stats>";
    static const int  statsz = sizeof(stats1) + (12*10) + 64;

    StatsData myData;

// If only the size is wanted, return the size
//
   if (!buff) return statsz;

// Make sure buffer is large enough
//
   if (blen < statsz) return 0;

// Get a copy of the statistics
//
   sdMutex.Lock();
   myData = Data;
   sdMutex.UnLock();

// Format the buffer
//
   return sprintf(buff, stats1, myRole, myData.numOpenR,   myData.numOpenW,
                    myData.numOpenP,    myData.numUnpsist, myData.numHandles,
                    myData.numRedirect, myData.numStarted, myData.numReplies,
                    myData.numErrors,   myData.numDelays,
                    myData.numSeventOK, myData.numSeventER);
}
