/******************************************************************************/
/*                                                                            */
/*                     X r d F r m R e q A g e n t . c c                      */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdFrmReqAgentCVSID = "$Id$";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdFrm/XrdFrmReqAgent.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdFrm/XrdFrmUtils.hh"
#include "XrdNet/XrdNetMsg.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysHeaders.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/
  
char *XrdFrmReqAgent::c2sFN = 0;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmReqAgent::XrdFrmReqAgent(const char *Me, int qVal)
              : Persona(Me),theQ(qVal)
{
// Set default ping message
//
   switch(qVal)
         {case XrdFrmRequest::getQ: pingMsg = "!<\n"; break;
          case XrdFrmRequest::migQ: pingMsg = "!&\n"; break;
          case XrdFrmRequest::stgQ: pingMsg = "!+\n"; break;
          case XrdFrmRequest::putQ: pingMsg = "!>\n"; break;
          default:                  pingMsg = "!\n" ; break;
         }
}

/******************************************************************************/
/* Public:                           A d d                                    */
/******************************************************************************/
  
void XrdFrmReqAgent::Add(XrdFrmRequest &Request)
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

// Now wake the boss
//
   Ping();
}

/******************************************************************************/
/* Public:                           D e l                                    */
/******************************************************************************/
  
void XrdFrmReqAgent::Del(XrdFrmRequest &Request)
{
   int i;
  
// Remove all pending requests for this id
//
   for (i = 0; i <= XrdFrmRequest::maxPrty; i++) rQueue[i]->Can(&Request);
}

/******************************************************************************/
/* Public:                          L i s t                                   */
/******************************************************************************/
  
void XrdFrmReqAgent::List(XrdFrmRequest::Item *Items, int Num)
{
   char myLfn[4096];
   int i, Offs;

// List entries in each priority queue
//
   for (i = 0; i <= XrdFrmRequest::maxPrty; i++)
       {Offs = 0;
        while(rQueue[i]->List(myLfn, sizeof(myLfn), Offs, Items, Num))
             cout <<myLfn <<endl;
       }
}
  
/******************************************************************************/
/* Public:                       N e x t L F N                                */
/******************************************************************************/
  
int XrdFrmReqAgent::NextLFN(char *Buff, int Bsz, int Prty, int &Offs)
{
   static XrdFrmRequest::Item Items[1] = {XrdFrmRequest::getLFN};

// Return entry, if it exists
//
   return rQueue[Prty]->List(Buff, Bsz, Offs, Items, 1) != 0;
}

/******************************************************************************/
/*                                  P i n g                                   */
/******************************************************************************/

void XrdFrmReqAgent::Ping(const char *Msg)
{
   static XrdNetMsg udpMsg(&Say, c2sFN);
   static int udpOK = 0;
   struct stat buf;

// Send given message or default message based on our persona
//
   if (udpOK || !stat(c2sFN, &buf))
      {udpMsg.Send(Msg ? Msg : pingMsg); udpOK = 1;}
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
int XrdFrmReqAgent::Start(char *aPath, int aMode)
{
   char buff[2048], *qPath;
   int i;

// Initialize the udp path for pings, if we have not done so
//
   if (!c2sFN)
      {sprintf(buff, "%sxfrd.udp", aPath);
       c2sFN = strdup(buff);
      }

// Generate the queue directory path
//
   if (!(qPath = XrdFrmUtils::makeQDir(aPath, aMode))) return 0;

// Initialize the request queues if all went well
//
   for (i = 0; i <= XrdFrmRequest::maxPrty; i++)
       {sprintf(buff, "%s%sQ.%d", qPath, Persona, i);
        rQueue[i] = new XrdFrmReqFile(buff, 1);
        if (!rQueue[i]->Init()) return 0;
       }

// All done
//
   return 1;
}
