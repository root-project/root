/******************************************************************************/
/*                                                                            */
/*                        X r d F r m P r o x y . c c                         */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

const char *XrdFrmProxyCVSID = "$Id$";

#include "errno.h"
#include "stdio.h"
#include "unistd.h"

#include "XrdFrm/XrdFrmReqAgent.hh"
#include "XrdFrm/XrdFrmProxy.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/
  
XrdFrmProxy::o2qMap XrdFrmProxy::oqMap[] =
                               {{"getf", XrdFrmRequest::getQ, opGet},
                                {"migr", XrdFrmRequest::migQ, opMig},
                                {"pstg", XrdFrmRequest::stgQ, opStg},
                                {"putf", XrdFrmRequest::putQ, opPut}};

int                 XrdFrmProxy::oqNum = sizeof(oqMap)/sizeof(oqMap[0]);

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmProxy::XrdFrmProxy(XrdSysLogger *lP, const char *iName, int Debug)
{
   char buff[256];

// Clear agent vector
//
   memset(Agent, 0, sizeof(Agent));

// Link the logger to our message facility
//
   Say.logger(lP);

// Set the debug flag
//
   if (Debug) Trace.What |= TRACE_ALL;

// Develop our internal name
//
   if (iName) insName = (strcmp(iName, "anon") ? iName : 0);
       insName = 0;
   sprintf(buff, "%s.%d", (iName ? iName : "anon"),static_cast<int>(getpid()));
   intName = strdup(buff);
}

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
int XrdFrmProxy::Add(char Opc, const char *Lfn, const char *Opq,
                               const char *Usr, const char *Rid,
                               const char *Nop, const char *Pop, int Prty)
{
   XrdFrmRequest myReq;
   int n, Options = 0;
   int qType = XrdFrmUtils::MapR2Q(Opc, &Options);

// Verify that we can support this operation
//
   if (!Agent[qType]) return -ENOTSUP;

// Initialize the request element
//
   memset(&myReq, 0, sizeof(myReq));
   myReq.OPc = Opc;

// Insert the Lfn and Opaque information
//
   n = strlen(Lfn);
   if (Opq && *Opq)
      {if (n + strlen(Opq) + 2 > sizeof(myReq.LFN)) return -ENAMETOOLONG;
       strcpy(myReq.LFN, Lfn); strcpy(myReq.LFN+n+1, Opq), myReq.Opaque = n;
      } else if (n < int(sizeof(myReq.LFN))) strcpy(myReq.LFN, Lfn);
                else return -ENAMETOOLONG;

// Get the LFN offset in case this is a url
//
   if (myReq.LFN[0] != '/' && !(myReq.LFO = XrdFrmUtils::chkURL(myReq.LFN)))
      return -EILSEQ;

// Set the user, request id, notification path, and priority
//
   if (Usr && *Usr) strlcpy(myReq.User, Usr, sizeof(myReq.User));
      else strcpy(myReq.User, intName);
   if (Rid) strlcpy(myReq.ID, Rid, sizeof(myReq.ID));
      else *(myReq.ID) = '?';
   if (Nop && *Nop) strlcpy(myReq.Notify, Nop, sizeof(myReq.Notify));
      else *(myReq.Notify) = '-';
   myReq.Prty = Prty;

// Establish processing options
//
   myReq.Options = Options | XrdFrmUtils::MapM2O(myReq.Notify, Pop);

// Add this request to the queue of requests via the agent
//
   Agent[qType]->Add(myReq);
   return 0;
}

/******************************************************************************/
/*                                   D e l                                    */
/******************************************************************************/
  
int XrdFrmProxy::Del(char Opc, const char *Rid)
{
   XrdFrmRequest myReq;
   int qType = XrdFrmUtils::MapR2Q(Opc);

// Verify that we can support this operation
//
   if (!Agent[qType]) return -ENOTSUP;

// Initialize the request element
//
   memset(&myReq, 0, sizeof(myReq));
   strlcpy(myReq.ID, Rid, sizeof(myReq.ID));

// Delete the request from the queue
//
   Agent[qType]->Del(myReq);
   return 0;
}

/******************************************************************************/
/*                                  L i s t                                   */
/******************************************************************************/
  
int XrdFrmProxy::List(XrdFrmProxy::Queues &State, char *Buff, int Bsz)
{
   int i;

// Get a queue type
//
do{if (!State.Active)
      while(State.QList & opAll)
           {for (i = 0; i < oqNum; i++) if (oqMap[i].oType & State.QList) break;
            if (i >= oqNum) return 0;
            State.QNow   =  oqMap[i].qType;
            State.QList &= ~oqMap[i].oType;
            if (!Agent[int(State.QNow)]) continue;
            State.Active = 1;
            break;
           }

   for (i = State.Prty; i <= XrdFrmRequest::maxPrty; i++)
       if (Agent[int(State.QNow)]->NextLFN(Buff,Bsz,i,State.Offset)) return 1;
          else State.Prty = i+1;

   State.Active = 0; State.Offset = 0; State.Prty = 0;
  } while(State.QList & opAll);

// We've completed returning all info
//
   return 0;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/

int XrdFrmProxy::Init(int opX, const char *aPath, int aMode)
{
   char *myAPath;
   int i;

// Create the admin directory if it does not exists
//
   if (!(myAPath = XrdFrmUtils::makePath(insName, aPath, aMode))) return 0;

// Now create and start an agent for each wanted service
//
   for (i = 0; i < oqNum; i++)
       if (opX & oqMap[i].oType)
          {Agent[oqMap[i].qType]
                = new XrdFrmReqAgent(oqMap[i].qName, oqMap[i].qType);
           if (!Agent[oqMap[i].qType]->Start(myAPath, aMode)) return 0;
          }

// All done
//
   return 1;
}
