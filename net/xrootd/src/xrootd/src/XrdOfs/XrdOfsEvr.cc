/******************************************************************************/
/*                                                                            */
/*                          X r d O f s E v r . c c                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdOfsEvrCVSID = "$Id$";

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "XrdCms/XrdCmsClient.hh"
#include "XrdOfs/XrdOfsEvr.hh"
#include "XrdOfs/XrdOfsStats.hh"
#include "XrdOfs/XrdOfsTrace.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysTimer.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucTrace.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdSys/XrdSysHeaders.hh"

/******************************************************************************/
/*                     E x t e r n a l   L i n k a g e s                      */
/******************************************************************************/

extern XrdOfsStats OfsStats;

extern XrdOucTrace OfsTrace;
  
void *XrdOfsEvRecv(void *pp)
{
     XrdOfsEvr *evr = (XrdOfsEvr *)pp;
     evr->recvEvents();
     return (void *)0;
}
  
void *XrdOfsEvFlush(void *pp)
{
     XrdOfsEvr *evr = (XrdOfsEvr *)pp;
     evr->flushEvents();
     return (void *)0;
}

int XrdOfsScrubScan(const char *key, XrdOfsEvr::theEvent *cip, void *xargp) 
    {return 0;}
  
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdOfsEvr::~XrdOfsEvr()
{

// Close the FIFO. This will cause the reader to exit
//
   myMutex.Lock();
   eventFIFO.Close();
   myMutex.UnLock();
}
  
/******************************************************************************/
/*                           f l u s h E v e n t s                            */
/******************************************************************************/
  
void XrdOfsEvr::flushEvents()
{
   theClient *tp, *ntp;
   int expWait, expClock;

// Compute the hash flush interval
//
   if ((expWait = maxLife/4) == 0) expWait = 60;
   expClock = expWait;

// We wait for the right period of time, unless there is a defered event
//
   do {myMutex.Lock(); 
       if ((ntp = deferQ)) deferQ = 0;
          else runQ = 0;
       myMutex.UnLock();
       while(ntp)
            {XrdSysTimer::Wait(1000*60);
             expClock -= 60;
             myMutex.Lock();
             while((tp = ntp))
                  {Events.Del(tp->Path);
                   ntp = tp->Next;
                   delete tp;
                  }
             if ((ntp = deferQ)) deferQ = 0;
                else runQ = 0;
             myMutex.UnLock();
             if (expClock <= 0)
                {myMutex.Lock(); 
                 Events.Apply(XrdOfsScrubScan, (void *)0);
                 myMutex.UnLock();
                 expClock = expWait;
                }
            }
       mySem.Wait();
      } while(1);
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdOfsEvr::Init(XrdSysError *eobj, XrdCmsClient *trgp)
{
   XrdNetSocket *msgSock;
   pthread_t     tid;
   int n, rc;
   char *p, *path, pbuff[2048];

// Set the error object and balancer pointers
//
   eDest    = eobj;
   Balancer = trgp;

// Create path to the pipe we will creat
//
   if (!(p = getenv("XRDADMINPATH")) || !*p)
      {eobj->Emsg("Events", "XRDADMINPATH not defined");
       return 0;
      }
   path = pbuff;
   strcpy(path, p); n = strlen(p);
   if (path[n-1] != '/') {path[n] = '/'; n++;}
   strcpy(&path[n], "ofsEvents");
   XrdOucEnv::Export("XRDOFSEVENTS", pbuff);

// Now create a socket to a path
//
   if (!(msgSock = XrdNetSocket::Create(eobj,path,0,0660,XRDNET_FIFO)))
      return 0;
   msgFD = msgSock->Detach();
   delete msgSock;

// Now start a thread to get incomming messages
//
   if ((rc = XrdSysThread::Run(&tid, XrdOfsEvRecv, static_cast<void *>(this),
                          0, "Event receiver")))
      {eobj->Emsg("Evr", rc, "create event reader thread");
       return 0;
      }

// Now start a thread to flush posted events
//
   if ((rc = XrdSysThread::Run(&tid, XrdOfsEvFlush,static_cast<void *>(this),
                          0, "Event flusher")))
      {eobj->Emsg("Evr", rc, "create event flush thread");
       return 0;
      }

// All done
//
   return 1;
}

/******************************************************************************/
/*                            r e c v E v e n t s                             */
/******************************************************************************/
  
void XrdOfsEvr::recvEvents()
{
   EPNAME("recvEvent");
   const char *tident = 0;
   char *lp,*tp;

// Attach the fifo FD to the stream
//
   eventFIFO.Attach(msgFD);

// Now just start reading the events until the FD is closed
//
   while((lp = eventFIFO.GetLine()))
        {DEBUG("-->" <<lp);
         if ((tp = eventFIFO.GetToken()) && *tp)
            {if (!strcmp(tp, "stage")) eventStage();
                else eDest->Emsg("Evr", "Unknown event name -", tp);
            }
        }
}
 
/******************************************************************************/
/*                            W a i t 4 E v e n t                             */
/******************************************************************************/
  
void XrdOfsEvr::Wait4Event(const char *path, XrdOucErrInfo *einfo)
{

// Replace original callback with our callback so we can queue this event
// after the wait request has been sent to the client. This avoids a race
// where the client might get the resume signal before the wait request.
//
   einfo->setErrCB((XrdOucEICB *)new theClient(this, einfo, path));
}
 
/******************************************************************************/
/*                            W o r k 4 E v e n t                             */
/******************************************************************************/
  
void XrdOfsEvr::Work4Event(theClient *Client)
{
   struct theEvent *anEvent;
   theClient *aClient = 0;

// First ste is to see if this event was posted
//
   myMutex.Lock();
   if (!(anEvent = Events.Find(Client->Path)))
      Events.Add(Client->Path, new theEvent(0, 0, Client), maxLife);
      else {aClient = anEvent->aClient;
            while(aClient)
                 {if (aClient->evtCB->Same(Client->evtCBarg,aClient->evtCBarg))
                     {aClient->evtCBarg = Client->evtCBarg;
                      break;
                     }
                  aClient = aClient->Next;
                 }
            if (!aClient) {Client->Next = anEvent->aClient;
                           anEvent->aClient = Client;
                          }
            if (anEvent->Happened) sendEvent(anEvent);
           }
   myMutex.UnLock();

// Delete the Client object if we really don't need it
//
   if (aClient) delete Client;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                            e v e n t S t a g e                             */
/******************************************************************************/
  
// stage {OK | ENOENT | BAD} <path> [<msg>] \n

void XrdOfsEvr::eventStage()
{
   int rc;
   char *tp, *eMsg, *altMsg = 0;
   struct theEvent *anEvent;

// Get the status token and decode it
//
   if (!(tp = eventFIFO.GetToken()))
      {eDest->Emsg("Evr", "Missing stage event status"); return;}

        if (!strcmp(tp, "OK"))     {rc = 0;
                                    OfsStats.Add(OfsStats.Data.numSeventOK);
                                   }
   else if (!strcmp(tp, "ENOENT")) {rc = ENOENT;
                                    altMsg = (char *)"file does not exist.";
                                   }
   else if (!strcmp(tp, "BAD"))    {rc = -1;
                                    OfsStats.Add(OfsStats.Data.numSeventOK);
                                    altMsg = (char *)"Dynamic staging failed.";
                                   }
   else {rc = -1;
         eDest->Emsg("Evr", "Invalid stage event status -", tp);
         altMsg = (char *)"Dynamic staging malfunctioned.";
         OfsStats.Add(OfsStats.Data.numSeventOK);
        }

// Get the path and optional message
//
   if (!(tp = eventFIFO.GetToken(&eMsg)))
      {eDest->Emsg("Evr", "Missing stage event path"); return;}
   if (rc)
      if (eMsg) {while(*eMsg == ' ') eMsg++;
                 if (!*eMsg) eMsg = altMsg;
                } else eMsg = altMsg;
      else eMsg = 0;

// At this point if we have a balancer, tell it what happened
//
   if (Balancer)
      {if (rc == 0) Balancer->Added(tp);
          else      Balancer->Removed(tp);
      }

// Either people are waiting for this event or it is preposted event.
//
   myMutex.Lock();
   if (!(anEvent = Events.Find(tp)))
      Events.Add(tp, new theEvent(rc, eMsg), maxLife);
      else {if (anEvent->finalRC == 0)
               {anEvent->finalRC = rc;
                if (eMsg) anEvent->finalMsg = strdup(eMsg);
                anEvent->Happened = 1;
               }
            if (anEvent->aClient) sendEvent(anEvent);
           }
   myMutex.UnLock();
}

/******************************************************************************/
/*                             s e n d E v e n t                              */
/******************************************************************************/
  
void XrdOfsEvr::sendEvent(theEvent *ep)
{
   theClient *cp;
   XrdOucErrInfo *einfo;
   int doDel = 0, Result = (ep->finalRC ? SFS_ERROR : SFS_OK);

// For each client, issue a call back sending the result back
// The event also goes in the defered delete queue as we need to hold on
// to it just in case a client is in-transit
//
   while((cp = ep->aClient))
        {einfo = new XrdOucErrInfo(cp->User, 0, cp->evtCBarg);
         einfo->setErrInfo(ep->finalRC, (ep->finalMsg ? ep->finalMsg : ""));
         cp->evtCB->Done(Result, einfo);
         ep->aClient = cp->Next;
         if (doDel) delete cp;
            else {cp->Next = deferQ; deferQ = cp; doDel = 1;}
        }

// Post the defer queue handler
//
   if (!runQ) {runQ = 1; mySem.Post();}
}
