/******************************************************************************/
/*                                                                            */
/*                       X r d C n s D a e m o n . c c                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCnsDaemonCVSID = "$Id$";

#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include "Xrd/XrdTrace.hh"

#include "XrdNet/XrdNetDNS.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOuc/XrdOucUtils.hh"

#include "XrdPosix/XrdPosixXrootd.hh"

#include "XrdCns/XrdCnsDaemon.hh"
#include "XrdCns/XrdCnsEvent.hh"

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

extern XrdSysError       XrdLog;

extern XrdOucTrace       XrdTrace;
 
extern XrdCnsDaemon      XrdCnsd;

       XrdPosixXrootd    XrdPosix;

/******************************************************************************/
/*                            d o R e q u e s t s                             */
/******************************************************************************/
  
void XrdCnsDaemon::doRequests()
{
   XrdCnsEvent *evP;
   unsigned char eType;

// Process requests as they come in
//
   do {evP = XrdCnsEvent::Remove(eType);
       switch (eType)
              {case XrdCnsEvent::evClosew: do_Trunc (evP); break;
               case XrdCnsEvent::evCreate: do_Create(evP); break;
               case XrdCnsEvent::evMkdir:  do_Mkdir (evP); break;
               case XrdCnsEvent::evMv:     do_Mv    (evP); break;
               case XrdCnsEvent::evRm:     do_Rm    (evP); break;
               case XrdCnsEvent::evRmdir:  do_Rmdir (evP); break;
               default: XrdLog.Emsg("doReq","Invalid event for", evP->Lfn1());
              }
       evP->Recycle();
      } while(1);
}

/******************************************************************************/
/*                             g e t E v e n t s                              */
/******************************************************************************/
  
void XrdCnsDaemon::getEvents(XrdOucStream &Events)
{
   const char *Miss = 0, *TraceID = "doEvents";
   long long Size;
   mode_t    Mode;
   const char *eP;
   char *tp, *etp;
   XrdCnsEvent *evP = 0;

// Each ofs request comes in as:
// <traceid> {closew <lfn> <size> | create <mode> <lfn> | mkdir <mode> <lfn> |
//            mv <lfn1> <lfn2>    | rm            <lfn> | rmdir        <lfn>}
//
   while((tp = Events.GetLine()))
        {TRACE(DEBUG, "Event: '" <<tp <<"'");
         eP = "?";
               if (!(tp = Events.GetToken()))          Miss = "traceid";
         else {evP = XrdCnsEvent::Alloc();
               if (!(eP = Events.GetToken())
               ||  !evP->setType(eP))                  Miss = "eventid";
         else {switch(evP->Type())
                     {case XrdCnsEvent::evClosew:
                           if (!(tp=Events.GetToken())) {Miss = "lfn";   break;}
                           evP->setLfn1(tp);
                           if (!(tp=Events.GetToken())) {Miss = "size";  break;}
                           Size = strtoll(tp, &etp, 10);
                           if (*etp)                    {Miss = "size";  break;}
                           evP->setSize(Size);
                           break;
                      case XrdCnsEvent::evCreate:
                      case XrdCnsEvent::evMkdir:
                           if (!(tp=Events.GetToken())) {Miss = "mode";  break;}
                           Mode = strtol(tp, &etp, 8);
                           if (*etp)                    {Miss = "mode";  break;}
                           evP->setMode(Mode);
                           if (!(tp=Events.GetToken())) {Miss = "lfn";   break;}
                           evP->setLfn1(tp);
                           break;
                      case XrdCnsEvent::evMv:
                           if (!(tp=Events.GetToken())) {Miss = "lfn1";  break;}
                           evP->setLfn1(tp);
                           if (!(tp=Events.GetToken())) {Miss = "lfn2";  break;}
                           evP->setLfn2(tp);
                           break;
                      default:     // rm | rmdir
                           if (!(tp=Events.GetToken())) {Miss = "lfn";   break;}
                           evP->setLfn1(tp);
                           break;
                     }
              } }

         if (Miss) {XrdLog.Emsg("doEvents", Miss, "missing in event", eP);
                    evP->Recycle();
                    continue;
                   }

         evP->Queue();
        }

// If we exit then we lost the connection
//
   XrdLog.Emsg("doEvents", "Exiting; lost event connection to xrootd!");
   exit(8);
}

/******************************************************************************/
/*                             d o _ C r e a t e                              */
/******************************************************************************/
  
void XrdCnsDaemon::do_Create(XrdCnsEvent *evP)
{
   int myFD;

// For now, simply open and create the file
//
   if ((myFD = XrdPosixXrootd::Open(evP->Lfn1(), O_WRONLY|O_CREAT, 0664)) >= 0)
      XrdPosixXrootd::Close(myFD);
      else XrdLog.Emsg("do_Create", errno, "create", evP->Lfn1());
}

/******************************************************************************/
/*                              d o _ M k d i r                               */
/******************************************************************************/
  
void XrdCnsDaemon::do_Mkdir(XrdCnsEvent *evP)
{
   if (XrdPosixXrootd::Mkdir(evP->Lfn1(), 0664))
      XrdLog.Emsg("do_Mkdir", errno, "mkdir", evP->Lfn1());
}

/******************************************************************************/
/*                                 d o _ M v                                  */
/******************************************************************************/
  
void XrdCnsDaemon::do_Mv(XrdCnsEvent *evP)
{
   if (XrdPosixXrootd::Rename(evP->Lfn1(), evP->Lfn2()))
      XrdLog.Emsg("do_Mv", errno, "mv", evP->Lfn1());
}

/******************************************************************************/
/*                                 d o _ R m                                  */
/******************************************************************************/
  
void XrdCnsDaemon::do_Rm(XrdCnsEvent *evP)
{
   if (XrdPosixXrootd::Unlink(evP->Lfn1()))
      XrdLog.Emsg("do_Rm", errno, "rm", evP->Lfn1());
}

/******************************************************************************/
/*                              d o _ R m d i r                               */
/******************************************************************************/
  
void XrdCnsDaemon::do_Rmdir(XrdCnsEvent *evP)
{
   if (XrdPosixXrootd::Rmdir(evP->Lfn1()))
      XrdLog.Emsg("do_Rmdir", errno, "rmdir", evP->Lfn1());
}

/******************************************************************************/
/*                              d o _ T r u n c                               */
/******************************************************************************/
  
void XrdCnsDaemon::do_Trunc(XrdCnsEvent *evP)
{
   int myFD;

// For now, simply open and trunc the file
//
   if ((myFD = XrdPosixXrootd::Open(evP->Lfn1(), O_WRONLY)) >= 0)
      {XrdPosixXrootd::Ftruncate(myFD, evP->Size());
       XrdPosixXrootd::Close(myFD);
      }
      else XrdLog.Emsg("do_Create", errno, "trunc", evP->Lfn1());
}
