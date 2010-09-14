/******************************************************************************/
/*                                                                            */
/*                      X r d F r m R e q B o s s . c c                       */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdFrmReqBossCVSID = "$Id$";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmCID.hh"
#include "XrdFrm/XrdFrmReqBoss.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdFrm/XrdFrmXfrQueue.hh"
#include "XrdNet/XrdNetMsg.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysHeaders.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                     T h r e a d   I n t e r f a c e s                      */
/******************************************************************************/
  
void *mainServerXeq(void *parg)
{
    XrdFrmReqBoss *theBoss = (XrdFrmReqBoss *)parg;
    theBoss->Process();
    return (void *)0;
}

/******************************************************************************/
/* Public:                           A d d                                    */
/******************************************************************************/
  
void XrdFrmReqBoss::Add(XrdFrmRequest &Request)
{

// Complete the request including verifying the priority
//
   if (Request.Prty > XrdFrmRequest::maxPrty)
      Request.Prty = XrdFrmRequest::maxPrty;
      else if (Request.Prty < 0)Request.Prty = 0;
   Request.addTOD = time(0);

// Now add it to the queue
//
   rQueue[static_cast<int>(Request.Prty)]->Add(&Request);

// Now wake ourselves up
//
   Wakeup(1);
}

/******************************************************************************/
/* Public:                           D e l                                    */
/******************************************************************************/
  
void XrdFrmReqBoss::Del(XrdFrmRequest &Request)
{
   int i;
  
// Remove all pending requests for this id
//
   for (i = 0; i <= XrdFrmRequest::maxPrty; i++) rQueue[i]->Can(&Request);
}

/******************************************************************************/
/* Public:                       P r o c e s s                                */
/******************************************************************************/
  
void XrdFrmReqBoss::Process()
{
   EPNAME("Process");
   XrdFrmRequest myReq;
   int i, rc, numXfr, numPull;;

// Perform staging in an endless loop
//
do{Wakeup(0);
   do{numXfr = 0;
      for (i = XrdFrmRequest::maxPrty; i >= 0; i--)
          {numPull = i+1;
           while(numPull && (rc = rQueue[i]->Get(&myReq)))
                {if (myReq.Options & XrdFrmRequest::Register) Register(myReq,i);
                    else {numPull -= XrdFrmXfrQueue::Add(&myReq,rQueue[i],theQ);
                          numXfr++;
                          DEBUG(Persona <<" from Q " << i <<' ' <<numPull <<" left");
                          if (rc < 0) break;
                         }
                }
          }
     } while(numXfr);
  } while(1);
}

/******************************************************************************/
/* Private:                     R e g i s t e r                               */
/******************************************************************************/

void XrdFrmReqBoss::Register(XrdFrmRequest &Req, int qNum)
{
   EPNAME("Register");
   char *eP;
   int Pid;

// Ignore this request if there is no cluster id or the process if is invalid
//
   if (!(*Req.LFN)) return;
   Pid = strtol(Req.ID, &eP, 10);
   if (*eP || Pid == 0) return;

// Register this cluster
//
   if (CID.Add(Req.iName, Req.LFN, static_cast<time_t>(Req.addTOD), Pid))
      {DEBUG("Instance=" <<Req.iName <<" cluster=" <<Req.LFN <<" pid=" <<Pid);}
      else rQueue[qNum]->Del(&Req);
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
int XrdFrmReqBoss::Start(char *aPath, int aMode)
{
   pthread_t tid;
   char buff[2048], *qPath;
   int retc, i;

// Generate the queue directory path
//
   if (!(qPath = XrdFrmUtils::makeQDir(aPath, aMode))) return 0;

// Initialize the request queues if all went well
//
   for (i = 0; i <= XrdFrmRequest::maxPrty; i++)
       {sprintf(buff, "%s%sQ.%d", qPath, Persona, i);
        rQueue[i] = new XrdFrmReqFile(buff, 0);
        if (!rQueue[i]->Init()) return 0;
       }

// Start the request processing thread
//
   if ((retc = XrdSysThread::Run(&tid, mainServerXeq, (void *)this,
                                 XRDSYSTHREAD_BIND, Persona)))
      {sprintf(buff, "create %s request thread", Persona);
       Say.Emsg("Start", retc, buff);
       return 0;
      }

// All done
//
   return 1;
}

/******************************************************************************/
/* Public:                        W a k e u p                                 */
/******************************************************************************/
  
void XrdFrmReqBoss::Wakeup(int PushIt)
{
   static XrdSysMutex     rqMutex;

// If this is a PushIt then see if we need to push the binary semaphore
//
   if (PushIt) {rqMutex.Lock();
                if (!isPosted) {rqReady.Post(); isPosted = 1;}
                rqMutex.UnLock();
               }
      else     {rqReady.Wait();
                rqMutex.Lock(); isPosted = 0; rqMutex.UnLock();
               }
}
