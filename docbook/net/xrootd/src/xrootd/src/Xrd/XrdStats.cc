/******************************************************************************/
/*                                                                            */
/*                           X r d S t a t s . c c                            */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$ 

const char *XrdStatsCVSID = "$Id$";

#if !defined(__macos__) && !defined(__FreeBSD__)
#include <malloc.h>
#endif
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
  
#include "XrdVersion.hh"
#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdJob.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdPoll.hh"
#include "Xrd/XrdProtLoad.hh"
#include "Xrd/XrdScheduler.hh"
#include "Xrd/XrdStats.hh"
#include "XrdNet/XrdNetMsg.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysTimer.hh"

/******************************************************************************/
/*           G l o b a l   C o n f i g u r a t i o n   O b j e c t            */
/******************************************************************************/

extern XrdBuffManager    XrdBuffPool;

extern XrdScheduler      XrdSched;

       long              XrdStats::tBoot = static_cast<long>(time(0));

/******************************************************************************/
/*               L o c a l   C l a s s   X r d S t a t s J o b                */
/******************************************************************************/
  
class XrdStatsJob : XrdJob
{
public:

     void DoIt() {Stats->Report();
                  XrdSched.Schedule((XrdJob *)this, time(0)+iVal);
                 }

          XrdStatsJob(XrdStats *sP, int iV) : XrdJob("stats reporter"),
                                              Stats(sP), iVal(iV)
                     {XrdSched.Schedule((XrdJob *)this, time(0)+iVal);}
         ~XrdStatsJob() {}
private:
XrdStats *Stats;
int       iVal;
};

/******************************************************************************/
/*                           C o n s t r c u t o r                            */
/******************************************************************************/
  
XrdStats::XrdStats(const char *hname, int port,
                   const char *iname, const char *pname)
{
   static const char *head =
          "<statistics tod=\"%%ld\" ver=\"" XrdVSTRING "\" src=\"%s:%d\" "
                      "tos=\"%ld\" pgm=\"%s\" ins=\"%s\" pid=\"%d\">";
   char myBuff[1024];

   Hlen = sprintf(myBuff, head, hname, port, tBoot, pname, iname,
                          static_cast<int>(getpid()));
   Head = strdup(myBuff);
   buff = 0;
   blen = 0;
   myHost = hname;
   myName = iname;
   myPort = port;
}
 
/******************************************************************************/
/*                                R e p o r t                                 */
/******************************************************************************/
  
void XrdStats::Report(char **Dest, int iVal, int Opts)
{
   extern XrdSysError XrdLog;
   static XrdNetMsg *netDest[2] = {0,0};
   static int autoSync, repOpts = Opts;
   XrdJob *jP;
   const char *Data;
          int theOpts, Dlen;

// If we have dest then this is for initialization
//
   if (Dest)
   // Establish up to two destinations
   //
      {if (Dest[0]) netDest[0] = new XrdNetMsg(&XrdLog, Dest[0]);
       if (Dest[1]) netDest[1] = new XrdNetMsg(&XrdLog, Dest[1]);
       if (!(repOpts & XRD_STATS_ALL)) repOpts |= XRD_STATS_ALL;
       autoSync = repOpts & XRD_STATS_SYNCA;

   // Get and schedule a new job to report (ignore the jP pointer afterwards)
   //
      if (netDest[0]) jP = (XrdJob *)new XrdStatsJob(this, iVal);
       return;
      }

// This is a re-entry for reporting purposes, establish the sync flag
//
   if (!autoSync || XrdSched.Active() <= 30) theOpts = repOpts;
      else theOpts = repOpts & ~XRD_STATS_SYNC;

// Now get the statistics
//
   Lock();
   if ((Data = Stats(theOpts)))
      {Dlen = strlen(Data);
       netDest[0]->Send(Data, Dlen);
       if (netDest[1]) netDest[1]->Send(Data, Dlen);
      }
   UnLock();
}

/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/
  
const char *XrdStats::Stats(int opts)   // statsMutex must be locked!
{
   static const char *sgen = "<stats id=\"sgen\">"
                             "<as>%d</as><et>%lu</et><toe>%ld</toe></stats>";
   static const char *tail = "</statistics>";
   static const char *snul = "<statistics tod=\"0\" ver=\"" XrdVSTRING "\">"
                            "</statistics>";

   static XrdProtLoad Protocols;
   static const int  ovrhed = 256+strlen(sgen)+strlen(tail);
   XrdSysTimer myTimer;
   char *bp;
   int   bl, sz, do_sync = (opts & XRD_STATS_SYNC ? 1 : 0);

// If buffer is not allocated, do it now. We must defer buffer allocation
// until all components that can provide statistics have been loaded
//
   if (!(bp = buff))
      {blen = InfoStats(0,0) + XrdBuffPool.Stats(0,0) + XrdLink::Stats(0,0)
            + ProcStats(0,0) + XrdSched.Stats(0,0)    + XrdPoll::Stats(0,0)
            + Protocols.Stats(0,0) + ovrhed + Hlen;
       buff = (char *)memalign(getpagesize(), blen+256);
       if (!(bp = buff)) return snul;
      }
   bl = blen;

// Start the time if need be
//
   if (opts & XRD_STATS_SGEN) myTimer.Reset();

// Insert the heading
//
   sz = sprintf(buff, Head, static_cast<long>(time(0)));
   bl -= sz; bp += sz;

// Extract out the statistics, as needed
//
   if (opts & XRD_STATS_INFO)
      {sz = InfoStats(bp, bl, do_sync);
       bp += sz; bl -= sz;
      }

   if (opts & XRD_STATS_BUFF)
      {sz = XrdBuffPool.Stats(bp, bl, do_sync);
       bp += sz; bl -= sz;
      }

   if (opts & XRD_STATS_LINK)
      {sz = XrdLink::Stats(bp, bl, do_sync);
       bp += sz; bl -= sz;
      }

   if (opts & XRD_STATS_POLL)
      {sz = XrdPoll::Stats(bp, bl, do_sync);
       bp += sz; bl -= sz;
      }

   if (opts & XRD_STATS_PROC)
      {sz = ProcStats(bp, bl, do_sync);
       bp += sz; bl -= sz;
      }

   if (opts & XRD_STATS_PROT)
      {sz = Protocols.Stats(bp, bl, do_sync);
       bp += sz; bl -= sz;
      }

   if (opts & XRD_STATS_SCHD)
      {sz = XrdSched.Stats(bp, bl, do_sync);
       bp += sz; bl -= sz;
      }

   if (opts & XRD_STATS_SGEN)
      {unsigned long totTime = 0;
       myTimer.Report(totTime);
       sz = snprintf(bp,bl,sgen,do_sync==0,totTime,static_cast<long>(time(0)));
       bp += sz; bl -= sz;
      }

   strlcpy(bp, tail, bl);
   return buff;
}
 
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                             I n f o S t a t s                              */
/******************************************************************************/
  
int XrdStats::InfoStats(char *bfr, int bln, int do_sync)
{
   static const char statfmt[] = "<stats id=\"info\"><host>%s</host>"
                     "<port>%d</port><name>%s</name></stats>";

// Check if actual length wanted
//
   if (!bfr) return sizeof(statfmt)+24 + strlen(myHost);

// Format the statistics
//
   return snprintf(bfr, bln, statfmt, myHost, myPort, myName);
}
 
/******************************************************************************/
/*                             P r o c S t a t s                              */
/******************************************************************************/
  
int XrdStats::ProcStats(char *bfr, int bln, int do_sync)
{
   static const char statfmt[] = "<stats id=\"proc\">"
          "<usr><s>%lld</s><u>%lld</u></usr>"
          "<sys><s>%lld</s><u>%lld</u></sys>"
          "</stats>";
   struct rusage r_usage;
   long long utime_sec, utime_usec, stime_sec, stime_usec;
// long long ru_maxrss, ru_majflt, ru_nswap, ru_inblock, ru_oublock;
// long long ru_msgsnd, ru_msgrcv, ru_nsignals;

// Check if actual length wanted
//
   if (!bfr) return sizeof(statfmt)+16*13;

// Get the statistics
//
   if (getrusage(RUSAGE_SELF, &r_usage)) return 0;

// Convert fields to correspond to the format we are using. Commented out fields
// are either not uniformaly reported or are incorrectly reported making them
// useless across multiple platforms.
//
//
   utime_sec   = static_cast<long long>(r_usage.ru_utime.tv_sec);
   utime_usec  = static_cast<long long>(r_usage.ru_utime.tv_usec);
   stime_sec   = static_cast<long long>(r_usage.ru_stime.tv_sec);
   stime_usec  = static_cast<long long>(r_usage.ru_stime.tv_usec);
// ru_maxrss   = static_cast<long long>(r_usage.ru_maxrss);
// ru_majflt   = static_cast<long long>(r_usage.ru_majflt);
// ru_nswap    = static_cast<long long>(r_usage.ru_nswap);
// ru_inblock  = static_cast<long long>(r_usage.ru_inblock);
// ru_oublock  = static_cast<long long>(r_usage.ru_oublock);
// ru_msgsnd   = static_cast<long long>(r_usage.ru_msgsnd);
// ru_msgrcv   = static_cast<long long>(r_usage.ru_msgrcv);
// ru_nsignals = static_cast<long long>(r_usage.ru_nsignals);

// Format the statistics
//
   return snprintf(bfr, bln, statfmt,
          utime_sec, utime_usec, stime_sec, stime_usec
//        ru_maxrss, ru_majflt, ru_nswap, ru_inblock, ru_oublock,
//        ru_msgsnd, ru_msgrcv, ru_nsignals
         );
}
