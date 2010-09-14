/******************************************************************************/
/*                                                                            */
/*                      X r d C m s M a n a g e r . c c                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// Original Version: 1.38 2007/07/26 15:18:24 ganis

const char *XrdCmsManagerCVSID = "$Id$";

#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/types.h>

#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsManager.hh"
#include "XrdCms/XrdCmsNode.hh"
#include "XrdCms/XrdCmsRouting.hh"
#include "XrdCms/XrdCmsTrace.hh"

#include "XrdSys/XrdSysTimer.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

       XrdCmsManager   XrdCms::Manager;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCmsManager::XrdCmsManager()
{
     memset((void *)MastTab, 0, sizeof(MastTab));
     MTHi    = -1;
}

/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
XrdCmsNode *XrdCmsManager::Add(XrdLink *lp, int Lvl)
{
   EPNAME("Add")
   XrdCmsNode *nP;
   int i;

// Find available ID for this node
//
   MTMutex.Lock();
   for (i = 0; i < MTMax; i++) if (!MastTab[i]) break;

// Check if we have too many here
//
   if (i >= MTMax)
      {MTMutex.UnLock();
       Say.Emsg("Manager", "Login to", lp->Name(), "failed; too many managers");
       return 0;
      }

// Obtain a new a new node object
//
   lp->setID("manager",0);
   if (!(nP = new XrdCmsNode(lp, 0, 0, Lvl, i)))
        {Say.Emsg("Manager", "Unable to obtain node object."); return 0;}

// Assign new manager
//
   MastTab[i] = nP;
   if (i > MTHi) MTHi = i;
   nP->isOffline  = 0;
   nP->isNoStage  = 0;
   nP->isSuspend  = 0;
   nP->isBound    = 1;
   nP->isConn     = 1;
   nP->isMan      = (Config.asManager() ? 1 : 0);
   MTMutex.UnLock();

// Document login
//
   DEBUG(nP->Name() <<" to manager config; id=" <<i);
   return nP;
}
  
/******************************************************************************/
/*                                I n f o r m                                 */
/******************************************************************************/
  
void XrdCmsManager::Inform(const char *What, const char *Data, int Dlen)
{
   EPNAME("Inform");
   XrdCmsNode *nP;
   int i;

// Obtain a lock on the table
//
   MTMutex.Lock();

// Run through the table looking for managers to send messages to
//
   for (i = 0; i <= MTHi; i++)
       {if ((nP=MastTab[i]) && !nP->isOffline)
           {nP->Lock();
            MTMutex.UnLock();
            DEBUG(nP->Name() <<" " <<What);
            nP->Send(Data, Dlen);
            nP->UnLock();
            MTMutex.Lock();
           }
       }
   MTMutex.UnLock();
}

/******************************************************************************/
  
void XrdCmsManager::Inform(const char *What, struct iovec *vP, int vN, int vT)
{
   EPNAME("Inform");
   int i;
   XrdCmsNode *nP;

// Obtain a lock on the table
//
   MTMutex.Lock();

// Run through the table looking for managers to send messages to
//
   for (i = 0; i <= MTHi; i++)
       {if ((nP=MastTab[i]) && !nP->isOffline)
           {nP->Lock();
            MTMutex.UnLock();
            DEBUG(nP->Name() <<" " <<What);
            nP->Send(vP, vN, vT);
            nP->UnLock();
            MTMutex.Lock();
           }
       }
   MTMutex.UnLock();
}
  
/******************************************************************************/

void XrdCmsManager::Inform(XrdCms::CmsReqCode rCode, int rMod,
                                  const char *Arg,  int Alen)
{
    CmsRRHdr Hdr = {0, rCode, rMod, htons(static_cast<unsigned short>(Alen))};
    struct iovec ioV[2] = {{(char *)&Hdr, sizeof(Hdr)},{(char *)Arg, Alen}};

    Inform(Router.getName((int)rCode), ioV, (Arg ? 2 : 1), Alen+sizeof(Hdr));
}

/******************************************************************************/

void XrdCmsManager::Inform(CmsRRHdr &Hdr, const char *Arg, int Alen)
{
    struct iovec ioV[2] = {{(char *)&Hdr, sizeof(Hdr)},{(char *)Arg, Alen}};

    Hdr.datalen = htons(static_cast<unsigned short>(Alen));

    Inform(Router.getName(Hdr.rrCode), ioV, (Arg ? 2 : 1), Alen+sizeof(Hdr));
}

/******************************************************************************/
/*                                R e m o v e                                 */
/******************************************************************************/

void XrdCmsManager::Remove(XrdCmsNode *nP, const char *reason)
{
   EPNAME("Remove")
   int sinst, sent = nP->ID(sinst);

// Obtain a lock on the servtab
//
   MTMutex.Lock();

// Make sure this node is the right one
//
   if (!(nP == MastTab[sent]))
      {MTMutex.UnLock();
       DEBUG("manager " <<sent <<'.' <<sinst <<" failed.");
       return;
      }

// Remove node from the manager table
//
   MastTab[sent] = 0;
   nP->isOffline = 1;
   DEBUG("completed " <<nP->Name() <<" manager " <<sent <<'.' <<sinst);

// Readjust MTHi
//
   if (sent == MTHi) while(MTHi >= 0 && !MastTab[MTHi]) MTHi--;
   MTMutex.UnLock();

// Document removal
//
   if (reason) Say.Emsg("Manager", nP->Ident, "removed;", reason);
}

/******************************************************************************/
/*                                 R e s e t                                  */
/******************************************************************************/
  
void XrdCmsManager::Reset()
{
   EPNAME("Reset");
   static CmsStatusRequest myState = {{0, kYR_status, 
                                       CmsStatusRequest::kYR_Reset, 0}};
   static const int        szReqst = sizeof(CmsStatusRequest);
   XrdCmsNode *nP;
   int i;

// Obtain a lock on the table
//
   MTMutex.Lock();

// Run through the table looking for managers to send a reset request
//
   for (i = 0; i <= MTHi; i++)
       {if ((nP=MastTab[i]) && !nP->isOffline && nP->isKnown)
           {nP->Lock();
            nP->isKnown = 0;
            MTMutex.UnLock();
            DEBUG("sent to " <<nP->Name());
            nP->Send((char *)&myState, szReqst);
            nP->UnLock();
            MTMutex.Lock();
           }
       }
   MTMutex.UnLock();
}
