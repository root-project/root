/******************************************************************************/
/*                                                                            */
/*                      X r d C m s C l u s t e r . c c                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/types.h>

#include "XProtocol/YProtocol.hh"
  
#include "Xrd/XrdJob.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdScheduler.hh"

#include "XrdCms/XrdCmsBaseFS.hh"
#include "XrdCms/XrdCmsCache.hh"
#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsCluster.hh"
#include "XrdCms/XrdCmsNode.hh"
#include "XrdCms/XrdCmsState.hh"
#include "XrdCms/XrdCmsSelect.hh"
#include "XrdCms/XrdCmsTrace.hh"
#include "XrdCms/XrdCmsTypes.hh"

#include "XrdNet/XrdNetDNS.hh"

#include "XrdOuc/XrdOucPup.hh"

#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysTimer.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

       XrdCmsCluster   XrdCms::Cluster;

/******************************************************************************/
/*                      L o c a l   S t r u c t u r e s                       */
/******************************************************************************/
  
class XrdCmsDrop : XrdJob
{
public:

     void DoIt() {Cluster.STMutex.Lock();
                  int rc = Cluster.Drop(nodeEnt, nodeInst, this);
                  Cluster.STMutex.UnLock();
                  if (!rc) delete this;
                 }

          XrdCmsDrop(int nid, int inst) : XrdJob("drop node")
                    {nodeEnt  = nid;
                     nodeInst = inst;
                     Sched->Schedule((XrdJob *)this, time(0)+Config.DRPDelay);
                    }
         ~XrdCmsDrop() {}

int  nodeEnt;
int  nodeInst;
};
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdCmsCluster::XrdCmsCluster()
{
     memset((void *)NodeTab, 0, sizeof(NodeTab));
     memset((void *)AltMans, (int)' ', sizeof(AltMans));
     cidFirst=  0;
     AltMend = AltMans;
     AltMent = -1;
     NodeCnt =  0;
     STHi    = -1;
     SelAcnt = 0;
     SelRcnt = 0;
     doReset = 0;
     resetMask = 0;
     peerHost  = 0;
     peerMask  = ~peerHost;
}
  
/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
XrdCmsNode *XrdCmsCluster::Add(XrdLink *lp, int port, int Status,
                               int sport, const char *theNID)
{
   EPNAME("Add")
   sockaddr InetAddr;
   const char *act = "";
   unsigned int ipaddr;
   XrdCmsNode *nP = 0;
   int Slot, Free = -1, Bump1 = -1, Bump2 = -1, Bump3 = -1, aSet = 0;
   int tmp, Special = (Status & (CMS_isMan|CMS_isPeer));
   XrdSysMutexHelper STMHelper(STMutex);

// Establish our IP address
//
   lp->Name(&InetAddr);
   ipaddr = XrdNetDNS::IPAddr(&InetAddr);

// Find available slot for this node. Here are the priorities:
// Slot  = Reconnecting node
// Free  = Available slot           ( 1st in table)
// Bump1 = Disconnected server      (last in table)
// Bump2 = Connected    server      (last in table) if new one is managr/peer
// Bump3 = Disconnected managr/peer ( 1st in table) if new one is managr/peer
//
   for (Slot = 0; Slot < STMax; Slot++)
       if (NodeTab[Slot])
          {if (NodeTab[Slot]->isNode(ipaddr, theNID)) break;
/*Conn*/   if (NodeTab[Slot]->isConn)
              {if (!NodeTab[Slot]->isPerm && Special)
                                             Bump2 = Slot; // Last conn Server
/*Disc*/      } else {
               if ( NodeTab[Slot]->isPerm)
                  {if (Bump3 < 0 && Special) Bump3 = Slot;}//  1st disc Man/Pr
                  else                       Bump1 = Slot; // Last disc Server
              }
          } else if (Free < 0)               Free  = Slot; //  1st free slot

// Check if node is already logged in or is a relogin
//
   if (Slot < STMax)
      {if (NodeTab[Slot] && NodeTab[Slot]->isBound)
          {Say.Emsg("Cluster", lp->ID, "already logged in.");
           return 0;
          } else { // Rehook node to previous unconnected entry
           nP = NodeTab[Slot];
           nP->Link      = lp;
           nP->isOffline = 0;
           nP->isConn    = 1;
           nP->Instance++;
           nP->setName(lp, port);  // Just in case it changed
           act = "Re-added ";
          }
      }

// Reuse an old ID if we must or redirect the incomming node
//
   if (!nP) 
      {if (Free >= 0) Slot = Free;
          else {if (Bump1 >= 0) Slot = Bump1;
                   else Slot = (Bump2 >= 0 ? Bump2 : Bump3);
                if (Slot < 0)
                   {if (Status & CMS_isPeer) Say.Emsg("Cluster", "Add peer", lp->ID,
                                                "failed; too many subscribers.");
                       else {sendAList(lp);
                             DEBUG(lp->ID <<" redirected; too many subscribers.");
                            }
                    return 0;
                   }

                if (Status & CMS_isMan) {setAltMan(Slot,ipaddr,sport); aSet=1;}
                if (NodeTab[Slot] && !(Status & CMS_isPeer))
                   sendAList(NodeTab[Slot]->Link);

                DEBUG(lp->ID << " bumps " << NodeTab[Slot]->Ident <<" #" <<Slot);
                NodeTab[Slot]->Lock();
                Remove("redirected", NodeTab[Slot], -1);
                act = "Shoved ";
               }
       NodeTab[Slot] = nP = new XrdCmsNode(lp, port, theNID, 0, Slot);
      }

// Indicate whether this snode can be redirected
//
   nP->isPerm = (Status & (CMS_isMan | CMS_isPeer)) ? CMS_Perm : 0;

// Assign new server
//
   if (!aSet && (Status & CMS_isMan)) setAltMan(Slot, ipaddr, sport);
   if (Slot > STHi) STHi = Slot;
   nP->isBound   = 1;
   nP->isConn    = 1;
   nP->isNoStage = (Status & CMS_noStage);
   nP->isSuspend = (Status & CMS_Suspend);
   nP->isMan     = (Status & CMS_isMan);
   nP->isPeer    = (Status & CMS_isPeer);
   nP->isDisable = 1;
   NodeCnt++;
   if (Config.SUPLevel
   && (tmp = NodeCnt*Config.SUPLevel/100) > Config.SUPCount)
      {Config.SUPCount=tmp; CmsState.Set(tmp);}

// Compute new peer mask, as needed
//
   if (nP->isPeer) peerHost |=  nP->NodeMask;
      else         peerHost &= ~nP->NodeMask;
   peerMask = ~peerHost;

// Assign a unique cluster number
//
   nP->myCNUM = Assign(nP->myCID);

// Document login
//
   DEBUG(act <<nP->Ident <<" to cluster " <<nP->myCNUM <<" slot "
         <<Slot <<'.' <<nP->Instance <<" (n=" <<NodeCnt <<" m="
         <<Config.SUPCount <<") ID=" <<nP->myNID);

// Compute new state of all nodes if we are a reporting manager.
//
   if (Config.asManager()) 
      CmsState.Update(XrdCmsState::Counts,nP->isSuspend?0:1,nP->isNoStage?0:1);

// All done
//
   return nP;
}

/******************************************************************************/
/*                             B r o a d c a s t                              */
/******************************************************************************/

SMask_t XrdCmsCluster::Broadcast(SMask_t smask, const struct iovec *iod,
                                 int iovcnt, int iotot)
{
   EPNAME("Broadcast")
   int i;
   XrdCmsNode *nP;
   SMask_t bmask, unQueried(0);

// Obtain a lock on the table and screen out peer nodes
//
   STMutex.Lock();
   bmask = smask & peerMask;

// Run through the table looking for nodes to send messages to
//
   for (i = 0; i <= STHi; i++)
       {if ((nP = NodeTab[i]) && nP->isNode(bmask))
           {nP->Lock();
            STMutex.UnLock();
            if (nP->Send(iod, iovcnt, iotot) < 0) 
               {unQueried |= nP->Mask();
                DEBUG(nP->Ident <<" is unreachable");
               }
            nP->UnLock();
            STMutex.Lock();
           }
       }
   STMutex.UnLock();
   return unQueried;
}

/******************************************************************************/

SMask_t XrdCmsCluster::Broadcast(SMask_t smask, XrdCms::CmsRRHdr &Hdr,
                                 char *Data,    int Dlen)
{
   struct iovec ioV[3], *iovP = &ioV[1];
   unsigned short Temp;
   int Blen;

// Construct packed data for the character argument. If data is a string then
// Dlen must include the null byte if it is specified at all.
//
   Blen  = XrdOucPup::Pack(&iovP, Data, Temp, (Dlen ? strlen(Data)+1 : Dlen));
   Hdr.datalen = htons(static_cast<unsigned short>(Blen));

// Complete the iovec and send off the data
//
   ioV[0].iov_base = (char *)&Hdr; ioV[0].iov_len = sizeof(Hdr);
   return Broadcast(smask, ioV, 3, Blen+sizeof(Hdr));
}

/******************************************************************************/

SMask_t XrdCmsCluster::Broadcast(SMask_t smask, XrdCms::CmsRRHdr &Hdr,
                                 void *Data,    int Dlen)
{
   struct iovec ioV[2] = {{(char *)&Hdr, sizeof(Hdr)}, {(char *)Data, Dlen}};

// Send of the data as eveything was constructed properly
//
   Hdr.datalen = htons(static_cast<unsigned short>(Dlen));
   return Broadcast(smask, ioV, 2, Dlen+sizeof(Hdr));
}

/******************************************************************************/
/*                             B r o a d s e n d                              */
/******************************************************************************/

int XrdCmsCluster::Broadsend(SMask_t Who, XrdCms::CmsRRHdr &Hdr, 
                             void *Data, int Dlen)
{
   EPNAME("Broadsend");
   static int Start = 0;
   XrdCmsNode *nP;
   struct iovec ioV[2] = {{(char *)&Hdr, sizeof(Hdr)}, {(char *)Data, Dlen}};
   int i, Beg, Fin, ioTot = Dlen+sizeof(Hdr);

// Send of the data as eveything was constructed properly
//
   Hdr.datalen = htons(static_cast<unsigned short>(Dlen));

// Obtain a lock on the table and get the starting and ending position. Note
// that the mechnism we use will necessarily skip newly added nodes.
//
   STMutex.Lock();
   Beg = Start = (Start <= STHi ? Start+1 : 0);
   Fin = STHi;

// Run through the table looking for a node to send the message to
//
do{for (i = Beg; i <= STHi; i++)
       {if ((nP = NodeTab[i]) && nP->isNode(Who))
           {nP->Lock();
            STMutex.UnLock();
            if (nP->Send(ioV, 2, ioTot) >= 0) {nP->UnLock(); return 1;}
            DEBUG(nP->Ident <<" is unreachable");
            nP->UnLock();
            STMutex.Lock();
           }
       }
    if (!Beg) break;
    Fin = Beg-1; Beg = 0;
   } while(1);

// Did not send to anyone
//
   STMutex.UnLock();
   return 0;
}
  
/******************************************************************************/
/*                               g e t M a s k                                */
/******************************************************************************/

SMask_t XrdCmsCluster::getMask(unsigned int IPv4adr)
{
   int i;
   XrdCmsNode *nP;
   SMask_t smask(0);

// Obtain a lock on the table
//
   STMutex.Lock();

// Run through the table looking for a node with matching IP address
//
   for (i = 0; i <= STHi; i++)
       if ((nP = NodeTab[i]) && nP->isNode(IPv4adr))
          {smask = nP->NodeMask; break;}

// All done
//
   STMutex.UnLock();
   return smask;
}

/******************************************************************************/

SMask_t XrdCmsCluster::getMask(const char *Cid)
{
   XrdCmsNode *nP;
   SMask_t smask(0);
   XrdOucTList *cP;
   int i = 1, Cnum = -1;

// Lock the cluster ID list
//
   cidMutex.Lock();

// Now find the cluster
//
   cP = cidFirst;
   while(cP && (i = strcmp(Cid, cP->text)) < 0) cP = cP->next;

// If we didn't find the cluster, return a mask of zeroes
//
   if (cP) Cnum = cP->val;
   cidMutex.UnLock();
   if (i) return smask;

// Obtain a lock on the table
//
   STMutex.Lock();

// Run through the table looking for a node with matching cluster number
//
   for (i = 0; i <= STHi; i++)
       if ((nP = NodeTab[i]) && nP->myCNUM == Cnum) smask |= nP->NodeMask;

// All done
//
   STMutex.UnLock();
   return smask;
}

/******************************************************************************/
/*                                  L i s t                                   */
/******************************************************************************/
  
XrdCmsSelected *XrdCmsCluster::List(SMask_t mask, CmsLSOpts opts)
{
    const char *reason;
    int i, iend, nump, delay, lsall = opts & LS_All;
    XrdCmsNode     *nP;
    XrdCmsSelected *sipp = 0, *sip;

// If only one wanted, the select appropriately
//
   STMutex.Lock();
   iend = (opts & LS_Best ? 0 : STHi);
   for (i = 0; i <= iend; i++)
       {if (opts & LS_Best)
            nP = (Config.sched_RR
                 ? SelbyRef( mask, nump, delay, &reason, 0)
                 : SelbyLoad(mask, nump, delay, &reason, 0));
           else if ((nP=NodeTab[i]) && !lsall && !(nP->NodeMask & mask)) nP=0;
        if (nP)
           {sip = new XrdCmsSelected((opts & LS_IPO) ? 0 : nP->Name(), sipp);
            if (opts & LS_IPV6)
               {sip->IPV6Len = nP->IPV6Len;
                strcpy(sip->IPV6, nP->IPV6);
               }
            sip->Mask    = nP->NodeMask;
            sip->Id      = nP->NodeID;
            sip->IPAddr  = nP->IPAddr;
            sip->Port    = nP->Port;
            sip->Load    = nP->myLoad;
            sip->Util    = nP->DiskUtil;
            sip->RefTotA = nP->RefTotA + nP->RefA;
            sip->RefTotR = nP->RefTotR + nP->RefR;
            if (nP->isOffline) sip->Status  = XrdCmsSelected::Offline;
               else sip->Status  = 0;
            if (nP->isDisable) sip->Status |= XrdCmsSelected::Disable;
            if (nP->isNoStage) sip->Status |= XrdCmsSelected::NoStage;
            if (nP->isSuspend) sip->Status |= XrdCmsSelected::Suspend;
            if (nP->isRW     ) sip->Status |= XrdCmsSelected::isRW;
            if (nP->isMan    ) sip->Status |= XrdCmsSelected::isMangr;
            if (nP->isPeer   ) sip->Status |= XrdCmsSelected::isPeer;
            if (nP->isProxy  ) sip->Status |= XrdCmsSelected::isProxy;
            nP->UnLock();
            sipp = sip;
           }
       }
   STMutex.UnLock();

// Return result
//
   return sipp;
}
  
/******************************************************************************/
/*                                L o c a t e                                 */
/******************************************************************************/

int XrdCmsCluster::Locate(XrdCmsSelect &Sel)
{
   EPNAME("Locate");
   XrdCmsPInfo   pinfo;
   SMask_t       qfVec(0);
   char         *Path;
   int           retc = 0;

// Check if this is a locate for all current servers
//
   if (*Sel.Path.Val != '*') Path = Sel.Path.Val;
      else {if (*(Sel.Path.Val+1) == '\0')
               {Sel.Vec.hf = ~0LL; Sel.Vec.pf = Sel.Vec.wf = 0;
                return 0;
               }
            Path = Sel.Path.Val+1;
           }

// Find out who serves this path
//
   if (!Cache.Paths.Find(Path, pinfo) || !pinfo.rovec)
      {Sel.Vec.hf = Sel.Vec.pf = Sel.Vec.wf = 0;
       return -1;
      } else Sel.Vec.wf = pinfo.rwvec;

// Check if this was a non-lookup request
//
   if (*Sel.Path.Val == '*')
      {Sel.Vec.hf = pinfo.rovec; Sel.Vec.pf = 0;
       Sel.Vec.wf = pinfo.rwvec;
       return 0;
      }

// Complete the request info object if we have one
//
   if (Sel.InfoP)
      {Sel.InfoP->rwVec = pinfo.rwvec;
       Sel.InfoP->isLU  = 1;
      }

// First check if we have seen this file before. If so, get nodes that have it.
// A Refresh request kills this because it's as if we hadn't seen it before.
// If the file was found but either a query is in progress or we have a server
// bounce; the client must wait.
//
   if (Sel.Opts & XrdCmsSelect::Refresh 
   || !(retc = Cache.GetFile(Sel, pinfo.rovec)))
      {Cache.AddFile(Sel, 0);
       qfVec = pinfo.rovec; Sel.Vec.hf = 0;
      } else qfVec = Sel.Vec.bf;

// Compute the delay, if any
//
   if ((!qfVec && retc >= 0) || (Sel.Vec.hf && Sel.InfoP)) retc =  0;
      else if (!(retc = Cache.WT4File(Sel, Sel.Vec.hf)))   retc = -2;

// Check if we have to ask any nodes if they have the file
//
   if (qfVec)
      {CmsStateRequest QReq = {{Sel.Path.Hash, kYR_state, kYR_raw, 0}};
       if (Sel.Opts & XrdCmsSelect::Refresh)
          QReq.Hdr.modifier |= CmsStateRequest::kYR_refresh;
       TRACE(Files, "seeking " <<Sel.Path.Val);
       qfVec = Cluster.Broadcast(qfVec, QReq.Hdr, 
                                 (void *)Sel.Path.Val, Sel.Path.Len+1);
       if (qfVec) Cache.UnkFile(Sel, qfVec);
      }
   return retc;
}
  
/******************************************************************************/
/*                               M o n P e r f                                */
/******************************************************************************/
  
void *XrdCmsCluster::MonPerf()
{
   CmsUsageRequest Usage = {{0, kYR_usage, 0, 0}};
   struct iovec ioV[] = {{(char *)&Usage, sizeof(Usage)}};
   int ioVnum = sizeof(ioV)/sizeof(struct iovec);
   int ioVtot = sizeof(Usage);
   SMask_t allNodes(~0);
   int uInterval = Config.AskPing*Config.AskPerf;

// Sleep for the indicated amount of time, then ask for load on each server
//
   while(uInterval)
        {XrdSysTimer::Snooze(uInterval);
         Broadcast(allNodes, ioV, ioVnum, ioVtot);
        }
   return (void *)0;
}
  
/******************************************************************************/
/*                               M o n R e f s                                */
/******************************************************************************/
  
void *XrdCmsCluster::MonRefs()
{
   XrdCmsNode *nP;
   int  i, snooze_interval = 10*60, loopmax, loopcnt = 0;
   int resetA, resetR, resetAR;

// Compute snooze interval
//
   if ((loopmax = Config.RefReset / snooze_interval) <= 1)
      {if (!Config.RefReset) loopmax = 0;
          else {loopmax = 1; snooze_interval = Config.RefReset;}
      }

// Sleep for the snooze interval. If a reset was requested then do a selective
// reset unless we reached our snooze maximum and enough selections have gone
// by; in which case, do a global reset.
//
   do {XrdSysTimer::Snooze(snooze_interval);
       loopcnt++;
       STMutex.Lock();
       resetA  = (SelAcnt >= Config.RefTurn);
       resetR  = (SelRcnt >= Config.RefTurn);
       resetAR = (loopmax && loopcnt >= loopmax && (resetA || resetR));
       if (doReset || resetAR)
           {for (i = 0; i <= STHi; i++)
                if ((nP = NodeTab[i])
                &&  (resetAR || (doReset && nP->isNode(resetMask))) )
                    {nP->Lock();
                     if (resetA || doReset) {nP->RefTotA += nP->RefA;nP->RefA=0;}
                     if (resetR || doReset) {nP->RefTotR += nP->RefR;nP->RefR=0;}
                     nP->UnLock();
                    }
            if (resetAR)
               {if (resetA) SelAcnt = 0;
                if (resetR) SelRcnt = 0;
                loopcnt = 0;
               }
            if (doReset) {doReset = 0; resetMask = 0;}
           }
       STMutex.UnLock();
      } while(1);
   return (void *)0;
}

/******************************************************************************/
/*                                R e m o v e                                 */
/******************************************************************************/

// Warning! The node object must be locked upon entry. The lock is released
//          prior to returning to the caller. This entry obtains the node
//          table lock. When immed != 0 then the node is immediately dropped.
//          When immed if < 0 then the caller already holds the STMutex and it 
//          is not released upon exit.

void XrdCmsCluster::Remove(const char *reason, XrdCmsNode *theNode, int immed)
{
   EPNAME("Remove_Node")
   struct theLocks
          {XrdSysMutex *myMutex;
           XrdCmsNode  *myNode;
           char        *myIdent;
           int          myImmed;
           int          myNID;
           int          myInst;

                       theLocks(XrdSysMutex *mtx, XrdCmsNode *node, int immed)
                               : myMutex(mtx), myNode(node), myImmed(immed)
                               {myIdent = strdup(node->Ident);
                                myNID = node->ID(myInst);
                                if (myImmed >= 0)
                                   {myNode->UnLock();
                                    myMutex->Lock();
                                    myNode->Lock();
                                   }
                               }
                      ~theLocks()
                               {if (myImmed >= 0) myMutex->UnLock();
                                if (myNode) myNode->UnLock();
                                free(myIdent);
                               }
          } LockHandler(&STMutex, theNode, immed);

   int Inst, NodeID = theNode->ID(Inst);

// The LockHandler makes sure that the proper locks are obtained in a deadlock
// free order. However, this may require that the node lock be released and
// then re-aquired. We check if we are still dealing with same node at entry.
// If not, issue message and high-tail it out.
//
   if (LockHandler.myNID != NodeID || LockHandler.myInst != Inst)
      {Say.Emsg("Manager", LockHandler.myIdent, "removal aborted.");
       DEBUG(LockHandler.myIdent <<" node " <<NodeID <<'.' <<Inst <<" != "
             << LockHandler.myNID <<'.' <<LockHandler.myInst <<" at entry.");
      }

// Mark node as being offline
//
   theNode->isOffline = 1;

// If the node is connected the simply close the connection. This will cause
// the connection handler to re-initiate the node removal. The LockHandler
// destructor will release the node table and node object locks as needed.
// This condition exists only if one node is being displaced by another node.
//
   if (theNode->isConn)
      {theNode->Disc(reason, 0);
       theNode->isGone = 1;
       return;
      }


// If the node is part of the cluster, do not count it anymore and
// indicate new state of this nodes if we are a reporting manager
//
   if (theNode->isBound)
      {theNode->isBound = 0; NodeCnt--;
       if (Config.asManager())
          CmsState.Update(XrdCmsState::Counts, theNode->isSuspend ? 0 : -1,
                                               theNode->isNoStage ? 0 : -1);
      }

// If this is an immediate drop request, do so now. Drop() will delete
// the node object and remove the node lock. So, tell LockHandler that.
//
   if (immed || !Config.DRPDelay) 
      {Drop(NodeID, Inst);
       LockHandler.myNode = 0;
       return;
      }

// If a drop job is already scheduled, update the instance field. Otherwise,
// Schedule a node drop at a future time.
//
   theNode->DropTime = time(0)+Config.DRPDelay;
   if (theNode->DropJob) theNode->DropJob->nodeInst = Inst;
      else theNode->DropJob = new XrdCmsDrop(NodeID, Inst);

// Document removal
//
   if (reason) 
      Say.Emsg("Manager", theNode->Ident, "scheduled for removal;", reason);
      else DEBUG(theNode->Ident <<" node " <<NodeID <<'.' <<Inst);
}

/******************************************************************************/
/*                              R e s e t R e f                               */
/******************************************************************************/
  
void XrdCmsCluster::ResetRef(SMask_t smask)
{

// Obtain a lock on the table
//
   STMutex.Lock();

// Inform the reset thread that we need a reset
//
   doReset = 1;
   resetMask |= smask;

// Unlock table and exit
//
   STMutex.UnLock();
}

/******************************************************************************/
/*                                S e l e c t                                 */
/******************************************************************************/
  
int XrdCmsCluster::Select(XrdCmsSelect &Sel)
{
   EPNAME("Select");
   XrdCmsPInfo  pinfo;
   const char  *Amode;
   int dowt = 0, retc, isRW, fRD, noSel = (Sel.Opts & XrdCmsSelect::Defer);
   SMask_t amask, smask, pmask;

// Establish some local options
//
   if (Sel.Opts & XrdCmsSelect::Write) 
      {isRW = 1; Amode = "write";
       if (Config.RWDelay)
          if (Sel.Opts & XrdCmsSelect::Create && Config.RWDelay < 2) fRD = 1;
             else fRD = 0;
          else fRD = 1;
      }
      else {isRW = 0; Amode = "read"; fRD = 1;}

// Find out who serves this path
//
   if (!Cache.Paths.Find(Sel.Path.Val, pinfo)
   || (amask = ((isRW ? pinfo.rwvec : pinfo.rovec) & ~Sel.nmask)) == 0)
      {Sel.Resp.DLen = snprintf(Sel.Resp.Data, sizeof(Sel.Resp.Data)-1,
                       "No servers have %s access to the file", Amode)+1;
       return -1;
      }

// If we are running a shared file system preform an optional restricted
// pre-selection and then do a standard selection.
//
   if (baseFS.isDFS())
      {pmask = amask;
       smask = (Sel.Opts & XrdCmsSelect::Online ? 0 : pinfo.ssvec & amask);
       if (baseFS.Trim())
          {Sel.Resp.DLen = 0;
           if (!(retc = SelDFS(Sel, amask, pmask, smask, isRW)))
              return (fRD ? Cache.WT4File(Sel,Sel.Vec.hf) : Config.LUPDelay);
           if (retc < 0) return -1;
          } else if (noSel) return 0;
       if ((pmask || smask) && (retc = SelNode(Sel, pmask, smask)) >= 0)
          return retc;
       Sel.Resp.DLen = snprintf(Sel.Resp.Data, sizeof(Sel.Resp.Data)-1,
                       "No servers are available to %s%s the file.",
                       Sel.Opts & XrdCmsSelect::Online ? "immediately " : "",
                       (smask ? "stage" : Amode))+1;
       return -1;
      }

// If either a refresh is wanted or we didn't find the file, re-prime the cache
// which will force the client to wait. Otherwise, compute the primary and
// secondary selections. If there are none, the client may have to wait if we
// have servers that we can query regarding the file. Note that for files being
// opened in write mode, only one writable copy may exist unless this is a
// meta-operation (e.g., remove) in which case the file itself remain unmodified
// or a replica request, in which case we select a new target server.
//
   if (!(Sel.Opts & XrdCmsSelect::Refresh)
   &&   (retc = Cache.GetFile(Sel, pinfo.rovec)))
      {if (isRW)
          {     if (Sel.Opts & XrdCmsSelect::Replica)
                   {pmask = amask & ~(Sel.Vec.hf | Sel.Vec.bf); smask = 0;
                    if (!pmask && !Sel.Vec.bf) return SelFail(Sel,eNoRep);
                   }
           else if (Sel.Vec.bf) pmask = smask = 0;
           else if (Sel.Vec.hf)
                   {if (Sel.Opts & XrdCmsSelect::NewFile) return SelFail(Sel,eExists);
                    if (!(Sel.Opts & XrdCmsSelect::isMeta)
                    &&  Multiple(Sel.Vec.hf))             return SelFail(Sel,eDups);
                    if (!(pmask = Sel.Vec.hf & amask))    return SelFail(Sel,eROfs);
                    smask = 0;
                   }
           else if (Sel.Opts & (XrdCmsSelect::Trunc | XrdCmsSelect::NewFile))
                   {pmask = amask; smask = 0;}
           else if ((smask = pinfo.ssvec & amask)) pmask = 0;
           else pmask = smask = 0;
          } else {
           pmask = Sel.Vec.hf  & amask; 
           if (Sel.Opts & XrdCmsSelect::Online) {pmask &= ~Sel.Vec.pf; smask=0;}
              else smask = pinfo.ssvec & amask;
          }
       if (Sel.Vec.hf & Sel.nmask) Cache.UnkFile(Sel, Sel.nmask);
      } else {
       Cache.AddFile(Sel, 0); 
       Sel.Vec.bf = pinfo.rovec; 
       Sel.Vec.hf = Sel.Vec.pf = pmask = smask = 0;
       retc = 0;
      }

// A wait is required if we don't have any primary or seconday servers
//
   dowt = (!pmask && !smask);

// If we can query additional servers, do so now. The client will be placed
// in the callback queue only if we have no possible selections
//
   if (Sel.Vec.bf)
      {CmsStateRequest QReq = {{Sel.Path.Hash, kYR_state, kYR_raw, 0}};
       if (Sel.Opts & XrdCmsSelect::Refresh)
          QReq.Hdr.modifier |= CmsStateRequest::kYR_refresh;
       if (dowt) retc= (fRD ? Cache.WT4File(Sel,Sel.Vec.hf) : Config.LUPDelay);
       TRACE(Files, "seeking " <<Sel.Path.Val);
       amask = Cluster.Broadcast(Sel.Vec.bf, QReq.Hdr,
                                 (void *)Sel.Path.Val,Sel.Path.Len+1);
       if (amask) Cache.UnkFile(Sel, amask);
       if (dowt) return retc;
      } else if (dowt && retc < 0 && !noSel)
                return (fRD ? Cache.WT4File(Sel,Sel.Vec.hf) : Config.LUPDelay);

// Broadcast a freshen up request if wanted
//
   if ((Sel.Opts & XrdCmsSelect::Freshen) && (amask = pmask & ~Sel.Vec.bf))
      {CmsStateRequest Qupt={{0,kYR_state,kYR_raw|CmsStateRequest::kYR_noresp,0}};
       Cluster.Broadcast(amask, Qupt.Hdr,(void *)Sel.Path.Val,Sel.Path.Len+1);
      }

// If we need to defer selection, simply return as this is a mindless prepare
//
   if (noSel) return 0;

// Select a node
//
   if (dowt || (retc = SelNode(Sel, pmask, smask)) < 0)
      {Sel.Resp.DLen = snprintf(Sel.Resp.Data, sizeof(Sel.Resp.Data)-1,
                       "No servers are available to %s%s the file.",
                       Sel.Opts & XrdCmsSelect::Online ? "immediately " : "",
                       (smask ? "stage" : Amode))+1;
       return -1;
      }

// All done
//
   return retc;
}

/******************************************************************************/
  
int XrdCmsCluster::Select(int isrw, SMask_t pmask,
                          int &port, char *hbuff, int &hlen)
{
   static const SMask_t smLow(255);
   XrdCmsNode *nP = 0;
   SMask_t tmask;
   const char *reason;
   int delay, nump, Snum = 0;

// If there is nothing to select from, return failure
//
   if (!pmask) return 0;

// If we are exporting a shared-everything system then the incomming mask
// may have more than one server indicated. So, we need to do a full select.
//
   if (baseFS.isDFS())
      {STMutex.Lock();
       nP = (Config.sched_RR
          ? SelbyRef( pmask, nump, delay, &reason, isrw)
          : SelbyLoad(pmask, nump, delay, &reason, isrw));
       STMutex.UnLock();
       if (!nP) return 0;
       strcpy(hbuff, nP->Name(hlen, port));
       nP->RefR++;
       nP->UnLock();
       return 1;
      }

// In shared-nothing systems the incomming mask will only have a single node.
// Compute the a single node number that is contained in the mask.
//
   do {if (!(tmask = pmask & smLow)) Snum += 8;
         else {while((tmask = tmask>>1)) Snum++; break;}
      } while((pmask = pmask >> 8));

// See if the node passes muster
//
   STMutex.Lock();
   if ((nP = NodeTab[Snum]))
      {if (nP->isOffline || nP->isSuspend || nP->isDisable)      nP = 0;
          else if (!Config.sched_RR
               && (nP->myLoad > Config.MaxLoad))                 nP = 0;
       if (nP)
          {if (isrw)
              if (nP->isNoStage || nP->DiskFree < nP->DiskMinF)  nP = 0;
                 else {SelAcnt++; nP->Lock();}
              else     {SelRcnt++; nP->Lock();}
          }
      }
   STMutex.UnLock();

// At this point either we have a node or we do not
//
   if (nP)
      {strcpy(hbuff, nP->Name(hlen, port));
       nP->RefR++;
       nP->UnLock();
       return 1;
      }
   return 0;
}

/******************************************************************************/
/*                               S e l F a i l                                */
/******************************************************************************/
  
int XrdCmsCluster::SelFail(XrdCmsSelect &Sel, int rc)
{
//
    const char *etext;

    switch(rc)
   {case eExists: etext = "Unable to create new file; file already exists.";
                  break;
    case eROfs:   etext = "Unable to write file; r/o file already exists.";
                  break;
    case eDups:   etext = "Unable to write file; multiple files exist.";
                  break;
    case eNoRep:  etext = "Unable to replicate file; no new sites available.";
                  break;
    default:      etext = "Unable to access file; file does not exist.";
                  break;
   };

    Sel.Resp.DLen = strlcpy(Sel.Resp.Data, etext, sizeof(Sel.Resp.Data))+1;
    return -1;
}
  
/******************************************************************************/
/*                                 S p a c e                                  */
/******************************************************************************/
  
void XrdCmsCluster::Space(SpaceData &sData, SMask_t smask)
{
   int i;
   XrdCmsNode *nP;
   SMask_t bmask;

// Obtain a lock on the table and screen out peer nodes
//
   STMutex.Lock();
   bmask = smask & peerMask;

// Run through the table getting space information
//
   for (i = 0; i <= STHi; i++)
       if ((nP = NodeTab[i]) && nP->isNode(bmask)
       &&  !nP->isOffline    && nP->isRW)
          {sData.Total += nP->DiskTotal;
           sData.sNum++;
           if (sData.sFree < nP->DiskFree)
              {sData.sFree = nP->DiskFree; sData.sUtil = nP->DiskUtil;}
           if (nP->isRW & XrdCmsNode::allowsRW)
              {sData.wNum++;
               if (sData.wFree < nP->DiskFree)
                  {sData.wFree = nP->DiskFree; sData.wUtil = nP->DiskUtil;
                   sData.wMinF = nP->DiskMinF;
                  }
              }
          }
   STMutex.UnLock();
}

/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/
  
int XrdCmsCluster::Stats(char *bfr, int bln)
{
   static const char statfmt1[] = "<stats id=\"cms\"><name>%s</name>";
   static const char statfmt2[] = "<subscriber><name>%s</name>"
          "<status>%s</status><load>%d</load><diskfree>%d</diskfree>"
          "<refa>%d</refa><refr>%d</refr></subscriber>";
   static const char statfmt3[] = "</stats>\n";
   XrdCmsSelected *sp;
   int mlen, tlen = sizeof(statfmt3);
   char stat[6], *stp;

   class spmngr {
         public: XrdCmsSelected *sp;

                 spmngr() {sp = 0;}
                ~spmngr() {XrdCmsSelected *xsp;
                           while((xsp = sp)) {sp = sp->next; delete xsp;}
                          }
                } mngrsp;

// Check if actual length wanted
//
   if (!bfr) return  sizeof(statfmt1) + 256  +
                    (sizeof(statfmt2) + 20*4 + 256) * STMax +
                     sizeof(statfmt3) + 1;

// Get the statistics
//
   mngrsp.sp = sp = List(FULLMASK, LS_All);

// Format the statistics
//
   mlen = snprintf(bfr, bln, statfmt1, Config.myName);
   if ((bln -= mlen) <= 0) return 0;
   tlen += mlen;

   while(sp && bln)
        {stp = stat;
         if (sp->Status)
            {if (sp->Status & XrdCmsSelected::Offline) *stp++ = 'o';
             if (sp->Status & XrdCmsSelected::Suspend) *stp++ = 's';
             if (sp->Status & XrdCmsSelected::NoStage) *stp++ = 'n';
             if (sp->Status & XrdCmsSelected::Disable) *stp++ = 'd';
            } else *stp++ = 'a';
         bfr += mlen;
         mlen = snprintf(bfr, bln, statfmt2, sp->Name, stat,
                sp->Load, sp->Free, sp->RefTotA, sp->RefTotR);
         bln  -= mlen;
         tlen += mlen;
         sp = sp->next;
        }

// See if we overflowed. otherwise finish up
//
   if (sp || bln < (int)sizeof(statfmt1)) return 0;
   bfr += mlen;
   strcpy(bfr, statfmt3);
   return tlen;
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                A s s i g n                                 */
/******************************************************************************/
  
int XrdCmsCluster::Assign(const char *Cid)
{
   static int cNum = 0;
   XrdOucTList *cP, *cPP, *cNew;
   int n = -1;

// Lock the cluster ID list
//
   cidMutex.Lock();

// Now find the cluster
//
   cP = cidFirst; cPP = 0;
   while(cP && (n = strcmp(Cid, cP->text)) < 0) {cPP = cP; cP = cP->next;}

// If an exiting cluster simply return the cluster number
//
   if (!n && cP) {n = cP->val; cidMutex.UnLock(); return n;}

// Add this cluster
//
   n = ++cNum;
   cNew = new XrdOucTList(Cid, cNum, cP);
   if (cPP) cPP->next = cNew;
      else  cidFirst  = cNew;

// Return the cluster number
//
   cidMutex.UnLock();
   return n;
}

/******************************************************************************/
/*                             c a l c D e l a y                              */
/******************************************************************************/
  
XrdCmsNode *XrdCmsCluster::calcDelay(int nump, int numd, int numf, int numo,
                                     int nums, int &delay, const char **reason)
{
        if (!nump) {delay = 0;
                    *reason = "no eligible servers for";
                   }
   else if (numf)  {delay = Config.DiskWT;
                    *reason = "no eligible servers have space for";
                   }
   else if (numo)  {delay = Config.MaxDelay;
                    *reason = "eligible servers overloaded for";
                   }
   else if (nums)  {delay = Config.SUSDelay;
                    *reason = "eligible servers suspended for";
                   }
   else if (numd)  {delay = Config.SUPDelay;
                    *reason = "eligible servers offline for";
                   }
   else            {delay = Config.SUPDelay;
                    *reason = "server selection error for";
                   }
   return (XrdCmsNode *)0;
}

/******************************************************************************/
/*                                  D r o p                                   */
/******************************************************************************/
  
// Warning: STMutex must be locked upon entry; the caller must release it.
//          This method may only be called via Remove() either directly or via
//          a defered job scheduled by that method. This method actually
//          deletes the node object.

int XrdCmsCluster::Drop(int sent, int sinst, XrdCmsDrop *djp)
{
   EPNAME("Drop_Node")
   XrdCmsNode *nP;
   char hname[512];

// Make sure this node is the right one
//
   if (!(nP = NodeTab[sent]) || nP->Inst() != sinst)
      {if (nP && djp == nP->DropJob) {nP->DropJob = 0; nP->DropTime = 0;}
       DEBUG(sent <<'.' <<sinst <<" cancelled.");
       return 0;
      }

// Check if the drop has been rescheduled
//
   if (djp && time(0) < nP->DropTime)
      {Sched->Schedule((XrdJob *)djp, nP->DropTime);
       return 1;
      }

// Save the node name (don't want to hold a lock across a message)
//
   strlcpy(hname, nP->Ident, sizeof(hname));

// Remove node from the node table
//
   NodeTab[sent] = 0;
   nP->isOffline = 1;
   nP->DropTime  = 0;
   nP->DropJob   = 0;
   nP->isBound   = 0;

// Remove node from the peer list (if it is one)
//
   if (nP->isPeer) {peerHost &= nP->NodeMask; peerMask = ~peerHost;}

// Remove node entry from the alternate list and readjust the end pointer.
//
   if (nP->isMan)
      {memset((void *)&AltMans[sent*AltSize], (int)' ', AltSize);
       if (sent == AltMent)
          {AltMent--;
           while(AltMent >= 0 &&  NodeTab[AltMent]
                              && !NodeTab[AltMent]->isMan) AltMent--;
           if (AltMent < 0) AltMend = AltMans;
              else AltMend = AltMans + ((AltMent+1)*AltSize);
          }
      }

// Readjust STHi
//
   if (sent == STHi) while(STHi >= 0 && !NodeTab[STHi]) STHi--;

// Invalidate any cached entries for this node
//
   if (nP->NodeMask) Cache.Drop(nP->NodeMask, sent, STHi);

// Document the drop
//
   Say.Emsg("Drop_Node", hname, "dropped.");

// Delete the node object
//
   delete nP;
   return 0;
}

/******************************************************************************/
/*                              M u l t i p l e                               */
/******************************************************************************/

int XrdCmsCluster::Multiple(SMask_t mVec)
{
   static const unsigned long long Left32  = 0xffffffff00000000LL;
   static const unsigned long long Right32 = 0x00000000ffffffffLL;
   static const unsigned long long Left16  = 0x00000000ffff0000LL;
   static const unsigned long long Right16 = 0x000000000000ffffLL;
   static const unsigned long long Left08  = 0x000000000000ff00LL;
   static const unsigned long long Right08 = 0x00000000000000ffLL;
   static const unsigned long long Left04  = 0x00000000000000f0LL;
   static const unsigned long long Right04 = 0x000000000000000fLL;
//                                0 1 2 3 4 5 6 7 8 9 A B C D E F
   static const int isMult[16] = {0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1};

   if (mVec & Left32) {if (mVec & Right32) return 1;
                          else mVec = mVec >> 32LL;
                      }
   if (mVec & Left16) {if (mVec & Right16) return 1;
                          else mVec = mVec >> 16LL;
                      }
   if (mVec & Left08) {if (mVec & Right08) return 1;
                          else mVec = mVec >>  8LL;
                      }
   if (mVec & Left04) {if (mVec & Right04) return 1;
                          else mVec = mVec >>  4LL;
                      }
   return isMult[mVec];
}
  
/******************************************************************************/
/*                                R e c o r d                                 */
/******************************************************************************/
  
void XrdCmsCluster::Record(char *path, const char *reason)
{
   EPNAME("Record")
   static int msgcnt = 256;
   static XrdSysMutex mcMutex;
   int mcnt;

   DEBUG(reason <<path);
   mcMutex.Lock();
   msgcnt++; mcnt = msgcnt;
   mcMutex.UnLock();

   if (mcnt > 255)
      {Say.Emsg("client defered;", reason, path);
       mcnt = 1;
      }
}
 
/******************************************************************************/
/*                               S e l N o d e                                */
/******************************************************************************/
  
int XrdCmsCluster::SelNode(XrdCmsSelect &Sel, SMask_t pmask, SMask_t amask)
{
    EPNAME("SelNode")
    const char *act=0, *reason, *reason2 = "";
    int pspace, needspace, delay = 0, delay2 = 0, nump, isalt = 0, pass = 2;
    SMask_t mask;
    XrdCmsNode *nP = 0;

// There is a difference bwteen needing space and needing r/w access. The former
// is needed when we will be writing data the latter for inode modifications.
//
   if (Sel.Opts & XrdCmsSelect::isMeta) needspace = 0;
      else needspace = (Sel.Opts & XrdCmsSelect::Write?XrdCmsNode::allowsRW:0);
   pspace = needspace;

// Scan for a primary and alternate node (alternates do staging). At this
// point we omit all peer nodes as they are our last resort.
//
   STMutex.Lock();
   mask = pmask & peerMask;
   while(pass--)
        {if (mask)
            {nP = (Config.sched_RR
                   ? SelbyRef( mask, nump, delay, &reason, needspace)
                   : SelbyLoad(mask, nump, delay, &reason, needspace));
             if (nP || (nump && delay) || NodeCnt < Config.SUPCount) break;
            }
         mask = amask & peerMask; isalt = XrdCmsNode::allowsSS;
         if (!(Sel.Opts & XrdCmsSelect::isMeta)) needspace |= isalt;
        }
   STMutex.UnLock();

// Update info
//
   if (nP)
      {strcpy(Sel.Resp.Data, nP->Name(Sel.Resp.DLen, Sel.Resp.Port));
       Sel.Resp.DLen++; Sel.smask = nP->NodeMask;
       if (isalt || (Sel.Opts & XrdCmsSelect::Create) || Sel.iovN)
          {if (isalt || (Sel.Opts & XrdCmsSelect::Create))
              {Sel.Opts |= (XrdCmsSelect::Pending | XrdCmsSelect::Advisory);
               if (Sel.Opts & XrdCmsSelect::noBind) act = " handling ";
                  else Cache.AddFile(Sel, nP->NodeMask);
              }
           if (Sel.iovN && Sel.iovP) 
              {nP->Send(Sel.iovP, Sel.iovN); act = " staging ";}
              else if (!act)                 act = " assigned ";
          } else                             act = " serving ";
       nP->UnLock();
       TRACE(Stage, Sel.Resp.Data <<act <<Sel.Path.Val);
       return 0;
      } else if (!delay && NodeCnt < Config.SUPCount)
                {reason = "insufficient number of nodes";
                 delay = Config.SUPDelay;
                }

// Return delay if selection failure is recoverable
//
   if (delay && delay < Config.PSDelay)
      {Record(Sel.Path.Val, reason);
       return delay;
      }

// At this point, we attempt a peer node selection (choice of last resort)
//
   if (Sel.Opts & XrdCmsSelect::Peers)
      {STMutex.Lock();
       if ((mask = (pmask | amask) & peerHost))
          nP = SelbyCost(mask, nump, delay2, &reason2, pspace);
       STMutex.UnLock();
       if (nP)
          {strcpy(Sel.Resp.Data, nP->Name(Sel.Resp.DLen, Sel.Resp.Port));
           Sel.Resp.DLen++; Sel.smask = nP->NodeMask;
           if (Sel.iovN && Sel.iovP) nP->Send(Sel.iovP, Sel.iovN);
           nP->UnLock();
           TRACE(Stage, "Peer " <<Sel.Resp.Data <<" handling " <<Sel.Path.Val);
           return 0;
          }
       if (!delay) {delay = delay2; reason = reason2;}
      }

// At this point we either don't have enough nodes or simply can't handle this
//
   if (delay)
      {TRACE(Defer, "client defered; " <<reason <<" for " <<Sel.Path.Val);
       return delay;
      }
   return -1;
}

/******************************************************************************/
/*                             S e l b y C o s t                              */
/******************************************************************************/

// Cost selection is used only for peer node selection as peers do not
// report a load and handle their own scheduling.

XrdCmsNode *XrdCmsCluster::SelbyCost(SMask_t mask, int &nump, int &delay,
                                     const char **reason, int needspace)
{
    int i, numd, numf, nums;
    XrdCmsNode *np, *sp = 0;

// Scan for a node (sp points to the selected one)
//
   nump = nums = numf = numd = 0; // possible, suspended, full, and dead
   for (i = 0; i <= STHi; i++)
       if ((np = NodeTab[i]) && (np->NodeMask & mask))
          {nump++;
           if (np->isOffline)                   {numd++; continue;}
           if (np->isSuspend || np->isDisable)  {nums++; continue;}
           if (needspace &&     np->isNoStage)  {numf++; continue;}
           if (!sp) sp = np;
              else if (abs(sp->myCost - np->myCost)
                          <= Config.P_fuzz)
                      {if (needspace)
                          {if (sp->RefA > (np->RefA+Config.DiskLinger))
                               sp=np;
                           } 
                           else if (sp->RefR > np->RefR) sp=np;
                       }
                       else if (sp->myCost > np->myCost) sp=np;
          }

// Check for overloaded node and return result
//
   if (!sp) return calcDelay(nump, numd, numf, 0, nums, delay, reason);
   sp->Lock();
   if (needspace) {SelAcnt++; sp->RefA++;}  // Protected by STMutex
      else        {SelRcnt++; sp->RefR++;}
   delay = 0;
   return sp;
}
  
/******************************************************************************/
/*                             S e l b y L o a d                              */
/******************************************************************************/
  
XrdCmsNode *XrdCmsCluster::SelbyLoad(SMask_t mask, int &nump, int &delay,
                                     const char **reason, int needspace)
{
    int i, numd, numf, numo, nums;
    int reqSS = needspace & XrdCmsNode::allowsSS;
    XrdCmsNode *np, *sp = 0;

// Scan for a node (preset possible, suspended, overloaded, full, and dead)
//
   nump = nums = numo = numf = numd = 0; 
   for (i = 0; i <= STHi; i++)
       if ((np = NodeTab[i]) && (np->NodeMask & mask))
          {nump++;
           if (np->isOffline)                     {numd++; continue;}
           if (np->isSuspend || np->isDisable)    {nums++; continue;}
           if (np->myLoad > Config.MaxLoad)       {numo++; continue;}
           if (needspace && (np->DiskFree < np->DiskMinF
                             || (reqSS && np->isNoStage)))
              {numf++; continue;}
           if (!sp) sp = np;
              else if (needspace)
                      {if (abs(sp->myMass - np->myMass) <= Config.P_fuzz)
                          {if (sp->RefA > (np->RefA+Config.DiskLinger)) sp=np;}
                          else if (sp->myMass > np->myMass)             sp=np;
                      } else {
                       if (abs(sp->myLoad - np->myLoad) <= Config.P_fuzz)
                          {if (sp->RefR > np->RefR)                     sp=np;}
                          else if (sp->myLoad > np->myLoad)             sp=np;
                      }
          }

// Check for overloaded node and return result
//
   if (!sp) return calcDelay(nump, numd, numf, numo, nums, delay, reason);
   sp->Lock();
   if (needspace) {SelAcnt++; sp->RefA++;}  // Protected by STMutex
      else        {SelRcnt++; sp->RefR++;}
   delay = 0;
   return sp;
}

/******************************************************************************/
/*                              S e l b y R e f                               */
/******************************************************************************/

XrdCmsNode *XrdCmsCluster::SelbyRef(SMask_t mask, int &nump, int &delay,
                                    const char **reason, int needspace)
{
    int i, numd, numf, nums;
    int reqSS = needspace & XrdCmsNode::allowsSS;
    XrdCmsNode *np, *sp = 0;

// Scan for a node (sp points to the selected one)
//
   nump = nums = numf = numd = 0; // possible, suspended, full, and dead
   for (i = 0; i <= STHi; i++)
       if ((np = NodeTab[i]) && (np->NodeMask & mask))
          {nump++;
           if (np->isOffline)                   {numd++; continue;}
           if (np->isSuspend || np->isDisable)  {nums++; continue;}
           if (needspace && (np->DiskFree < np->DiskMinF
                             || (reqSS && np->isNoStage)))
              {numf++; continue;}
           if (!sp) sp = np;
              else if (needspace)
                      {if (sp->RefA > (np->RefA+Config.DiskLinger)) sp=np;}
                      else if (sp->RefR > np->RefR) sp=np;
          }

// Check for overloaded node and return result
//
   if (!sp) return calcDelay(nump, numd, numf, 0, nums, delay, reason);
   sp->Lock();
   if (needspace) {SelAcnt++; sp->RefA++;}  // Protected by STMutex
      else        {SelRcnt++; sp->RefR++;}
   delay = 0;
   return sp;
}
 
/******************************************************************************/
/*                                S e l D F S                                 */
/******************************************************************************/
  
int XrdCmsCluster::SelDFS(XrdCmsSelect &Sel, SMask_t amask,
                          SMask_t &pmask, SMask_t &smask, int isRW)
{
   EPNAME("SelDFS");
   static const SMask_t allNodes(~0);
   int oldOpts, rc;

// The first task is to find out if the file exists somewhere. If we are doing
// local queries, then the answer will be immediate. Otherwise, forward it.
//
   if ((Sel.Opts & XrdCmsSelect::Refresh) || !(rc = Cache.GetFile(Sel, amask)))
      {if (!baseFS.Local())
          {CmsStateRequest QReq = {{Sel.Path.Hash, kYR_state, kYR_raw, 0}};
           TRACE(Files, "seeking " <<Sel.Path.Val);
           Cluster.Broadsend(amask, QReq.Hdr, Sel.Path.Val, Sel.Path.Len+1);
           return 0;
          }
       if ((rc = baseFS.Exists(Sel.Path.Val, -Sel.Path.Len)) < 0)
          {Cache.AddFile(Sel, 0);
           Sel.Vec.bf = Sel.Vec.pf = Sel.Vec.wf = Sel.Vec.hf = 0;
          } else {
           Sel.Vec.hf = amask; Sel.Vec.wf = (isRW ? amask : 0);
           oldOpts = Sel.Opts;
           if (rc != CmsHaveRequest::Pending) Sel.Vec.pf = 0;
              else {Sel.Vec.pf = amask; Sel.Opts |= XrdCmsSelect::Pending;}
           Cache.AddFile(Sel, allNodes);
           Sel.Opts = oldOpts;
          }
      }

// Screen out online requests where the file is pending
//
   if (Sel.Opts & XrdCmsSelect::Online && Sel.Vec.pf)
      {pmask = smask = 0;
       return 1;
      }

// If the file is to be written and the files exists then it can't be a new file
//
   if (isRW && Sel.Vec.hf)
      {if (Sel.Opts & XrdCmsSelect::NewFile) return SelFail(Sel,eExists);
       if (Sel.Opts & XrdCmsSelect::Trunc) smask = 0;
       return 1;
      }

// Final verification that we have something to select
//
   if (!Sel.Vec.hf && (!isRW || !(Sel.Opts & XrdCmsSelect::NewFile)))
      return SelFail(Sel, eNoEnt);
   return 1;
}
  
/******************************************************************************/
/*                             s e n d A L i s t                              */
/******************************************************************************/
  
// Single entry at a time, protected by STMutex!

void XrdCmsCluster::sendAList(XrdLink *lp)
{
   static CmsTryRequest Req = {{0, kYR_try, 0, 0}, 0};
   static int HdrSize = sizeof(Req.Hdr) + sizeof(Req.sLen);
   static char *AltNext = AltMans;
   static struct iovec iov[4] = {{(caddr_t)&Req, HdrSize},
                                 {0, 0},
                                 {AltMans, 0},
                                 {(caddr_t)"\0", 1}};
   int dlen;

// Calculate what to send
//
   AltNext = AltNext + AltSize;
   if (AltNext >= AltMend)
      {AltNext = AltMans;
       iov[1].iov_len = 0;
       iov[2].iov_len = dlen = AltMend - AltMans;
      } else {
        iov[1].iov_base = (caddr_t)AltNext;
        iov[1].iov_len  = AltMend - AltNext;
        iov[2].iov_len  = AltNext - AltMans;
        dlen = iov[1].iov_len + iov[2].iov_len;
      }

// Complete the request (account for trailing null character)
//
   dlen++;
   Req.Hdr.datalen = htons(static_cast<unsigned short>(dlen+sizeof(Req.sLen)));
   Req.sLen = htons(static_cast<unsigned short>(dlen));

// Send the list of alternates (rotated once)
//
   lp->Send(iov, 4, dlen+HdrSize);
}

/******************************************************************************/
/*                             s e t A l t M a n                              */
/******************************************************************************/
  
// Single entry at a time, protected by STMutex!
  
void XrdCmsCluster::setAltMan(int snum, unsigned int ipaddr, int port)
{
   char *ap = &AltMans[snum*AltSize];
   int i;

// Preset the buffer and pre-screen the port number
//
   if (!port || (port > 0x0000ffff)) port = Config.PortTCP;
   memset(ap, int(' '), AltSize);

// Insert the ip address of this node into the list of nodes
//
   i = XrdNetDNS::IP2String(ipaddr, port, ap, AltSize);
   ap[i] = ' ';

// Compute new fence
//
   if (ap >= AltMend) {AltMend = ap + AltSize; AltMent = snum;}
}
