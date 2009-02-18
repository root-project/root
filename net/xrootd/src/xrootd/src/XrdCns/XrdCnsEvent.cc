/******************************************************************************/
/*                                                                            */
/*                        X r d C n s E v e n t . c c                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdCnsEventCVSID = "$Id$";

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "XrdCns/XrdCnsEvent.hh"
 
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
extern XrdSysError                  XrdLog;

XrdSysMutex            XrdCnsEvent::aqMutex;
XrdSysMutex            XrdCnsEvent::dqMutex;
XrdSysSemaphore        XrdCnsEvent::mySem(0);
XrdCnsEvent           *XrdCnsEvent::freeEvent = 0;
XrdCnsEvent           *XrdCnsEvent::frstEvent = 0;
XrdCnsEvent           *XrdCnsEvent::lastEvent = 0;
char                   XrdCnsEvent::pfxPath[1024] = {'\0'};
char                   XrdCnsEvent::logFN[1024] = {'\0'};
unsigned int           XrdCnsEvent::EventNumber = 0;
int                    XrdCnsEvent::logFD = 0;
int                    XrdCnsEvent::logOffset = 0;
int                    XrdCnsEvent::logOffmax = 0;
int                    XrdCnsEvent::maxLen1   = lfnBSize;
int                    XrdCnsEvent::pfxLen = 0;
char                   XrdCnsEvent::Running = 0;

/******************************************************************************/
/*                                  A l l o c                                  */
/******************************************************************************/
  
XrdCnsEvent *XrdCnsEvent::Alloc()
{
   XrdCnsEvent *ep;

// Allocate a request object. If we have no memory, tell the requester
// to try again in a minute.
//
   aqMutex.Lock();
   if ((ep = freeEvent)) freeEvent = ep->Next;
      else ep = new XrdCnsEvent();
   EventNumber++;
   ep->Event.Number = EventNumber;
   aqMutex.UnLock();

   ep->Next        = 0;
   ep->EventLen    = 0;
   ep->Event.Size  = 0;
   return ep;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdCnsEvent::Init(const char *aP, const char *pP, int qLim)
{
   struct stat buf;
   char prvFN[1024];

// Set the logfile size
//
   logOffmax = qLim*sizeof(EventRec);

// Copy the path prefix
//
   strcpy(pfxPath, pP);
   pfxLen = strlen(pP);

// Construct the long file name
//
   strcpy(logFN, aP); strcat(logFN, "XrdCnsd.eventLog");
   strcpy(prvFN, aP); strcat(prvFN, "XrdCnsd.Recovery");

// If there is a recovery file then we will process it. Otherwise, if a log file
// already exists then it becomes the recovery file.
//
   if (!stat(prvFN, &buf))  unlink(logFN);
      else if (!stat(logFN, &buf))
              {if (rename(logFN, prvFN))
                  {XrdLog.Emsg("Event", errno, "rename", logFN); return 0;}
              }
              else buf.st_size = 0;

// Open the event log file (creating if necessary). Recover() will adjust the
// starting offset.
//
   if ((logFD = open(logFN, O_RDWR|O_CREAT, 0664)) < 0)
      {XrdLog.Emsg("Event", errno, "open", logFN); return 0;}

// Perform recovery if we have something to recover from
//
   if (buf.st_size && !Recover(prvFN, buf.st_size)) return 0;

// All done
//
   return 1;
}

/******************************************************************************/
/*                                 Q u e u e                                  */
/******************************************************************************/
  
void XrdCnsEvent::Queue()
{

// Log this request
//
   dqMutex.Lock();
   if (pwrite(logFD,&Event,EventLen,logOffset) < 0 || fsync(logFD) < 0) return;
   EventOff  = logOffset;
   logOffset = (logOffset + sizeof(Event)) % logOffmax;

// Place it on the processing queue
//
   if (frstEvent) lastEvent->Next = this;
      else        frstEvent       = this;
   lastEvent = this;

// Tell dequeue thread we have something if it's not already running
//
   if (!Running) {mySem.Post(); Running = 1;}
   dqMutex.UnLock();
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdCnsEvent::Recycle()
{
   static unsigned long long Zero = 0;

// Clear the slot entry by writing a zero there
//
   dqMutex.Lock();
   if (pwrite(logFD, &Zero, sizeof(Zero), EventOff) >= 0) fsync(logFD);
   dqMutex.UnLock();

// Put this object on the free queue
//
   aqMutex.Lock();
   Next = freeEvent;
   freeEvent = this;
   aqMutex.UnLock();
}

/******************************************************************************/
/*                                R e m o v e                                 */
/******************************************************************************/
  
XrdCnsEvent *XrdCnsEvent::Remove(unsigned char &eT)
{
   XrdCnsEvent *ep;

// Find the request in the slot table
//
   dqMutex.Lock();

   while(!(ep = frstEvent))
        {Running = 0;
         dqMutex.UnLock();
         mySem.Wait();
         dqMutex.Lock();
        }

// Fix up queue and return the event
//
   frstEvent = ep->Next;
   dqMutex.UnLock();
   eT = ep->Event.Type;
   return ep;
}

/******************************************************************************/
/*                               s e t T y p e                                */
/******************************************************************************/

int XrdCnsEvent::setType(const char *eType)
{
        if (!strcmp(eType, "closew")) setType(evClosew);
   else if (!strcmp(eType, "create")) setType(evCreate);
   else if (!strcmp(eType, "mkdir"))  setType(evMkdir);
   else if (!strcmp(eType, "mv"))     setType(evMv);
   else if (!strcmp(eType, "rm"))     setType(evRm);
   else if (!strcmp(eType, "rmdir"))  setType(evRmdir);
   else return 0;

   return 1;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               R e c o v e r                                */
/******************************************************************************/
  
int XrdCnsEvent::Recover(const char *theFN, off_t theSize)
{
   static const off_t minSz = HdrSize;
   static const size_t evSz = sizeof(EventRec);
   struct EventRec myEvent;
   XrdCnsEvent *evP, *frstWrap = 0, *lastWrap = 0;
   unsigned int lastRnum = 0;
   int n, theFD, recs = 0;

// Verify that the size of the file is correct
//
   n = theSize % sizeof(EventRec);
   if (n && n < minSz)
      {XrdLog.Emsg("Event", "Damaged or invalid", theFN); return 0;}

// Open the file
//
   if ((theFD = open(theFN, O_RDWR)) < 0)
      {XrdLog.Emsg("Event", errno, "open", theFN); return 0;}

// Recover all of the items in the log
//
   while((n = read(theFD, &myEvent, evSz)) >= ssize_t(minSz))
        {if (!myEvent.Type) continue;
         evP = Alloc(); evP->Event = myEvent; recs++;
         if (frstWrap || myEvent.Number < lastRnum)
            {if (frstWrap ) lastWrap ->Next = evP;
                else        frstWrap        = evP;
             lastWrap  = evP;
            } else {
             if (frstEvent) lastEvent->Next = evP;
                else        frstEvent       = evP;
             lastEvent = evP;
            }
         }

// Verify all went well
//
   if (n && n < int(minSz))
      {XrdLog.Emsg("Event", errno, "read", theFN); return 0;}

// Rechain the log if it wrapped
//
   if (frstWrap)
      {lastWrap->Next = frstEvent;
       frstEvent      = frstWrap;
      }

// Now rewrite all of the events back into a new log file renumbering
// all of the events.
//
   evP = frstEvent;
   while(evP)
        {evP->Event.Number = ++EventNumber;
         evP->EventOff     = logOffset;
         if (pwrite(theFD, evP, evSz, logOffset)
             != sizeof(EventRec))
            {XrdLog.Emsg("Event", errno, "write", logFN); return 0;}
         logOffset += sizeof(EventRec);
         evP = evP->Next;
        }

// Sync the file and remove the recovery file
//
   fsync(logFD);
   close(theFD);
   unlink(theFN);

// Indicate how many we recovered and return
//
   if (recs)
      {char buff[256];
       sprintf(buff, "Recovered %d event(s) from", recs);
       XrdLog.Emsg("Event", buff, logFN);
      }
   return 1;
}
  
