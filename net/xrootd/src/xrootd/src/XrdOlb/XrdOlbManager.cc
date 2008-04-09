/******************************************************************************/
/*                                                                            */
/*                      X r d O l b M a n a g e r . c c                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOlbManagerCVSID = "$Id$";

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
  
#include "Xrd/XrdJob.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdOlb/XrdOlbCache.hh"
#include "XrdOlb/XrdOlbConfig.hh"
#include "XrdOlb/XrdOlbManager.hh"
#include "XrdOlb/XrdOlbManList.hh"
#include "XrdOlb/XrdOlbManTree.hh"
#include "XrdOlb/XrdOlbRTable.hh"
#include "XrdOlb/XrdOlbServer.hh"
#include "XrdOlb/XrdOlbState.hh"
#include "XrdOlb/XrdOlbTrace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdNet/XrdNetWork.hh"

using namespace XrdOlb;

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

       XrdOlbManager   XrdOlb::Manager;

/******************************************************************************/
/*                      L o c a l   S t r u c t u r e s                       */
/******************************************************************************/
  
class XrdOlbDrop : XrdJob
{
public:

     void DoIt() {Manager.STMutex.Lock();
                  Manager.Drop_Server(servEnt, servInst, this);
                 }

          XrdOlbDrop(int sid, int inst) : XrdJob("drop server")
                    {servEnt  = sid;
                     servInst = inst;
                     Sched->Schedule((XrdJob *)this, time(0)+Config.DRPDelay);
                    }
         ~XrdOlbDrop() {}

int  servEnt;
int  servInst;
};
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdOlbManager::XrdOlbManager()
{
     memset((void *)ServTab, 0, sizeof(ServTab));
     memset((void *)ServBat, 0, sizeof(ServBat));
     memset((void *)MastTab, 0, sizeof(MastTab));
     memset((void *)AltMans, (int)' ', sizeof(AltMans));
     AltMend = AltMans;
     AltMent = -1;
     ServCnt =  0;
     hasData =  0;
     MTHi    = -1;
     STHi    = -1;
     XWait   = 0;
     XnoStage= 0;
     SelAcnt = 0;
     SelRcnt = 0;
     doReset = 0;
     resetMask = 0;
     peerHost  = 0;
     peerMask  = 0; peerMask = ~peerMask;
}
  
/******************************************************************************/
/*                             B r o a d c a s t                              */
/******************************************************************************/
  
void XrdOlbManager::Broadcast(SMask_t smask, char *buff, int blen)
{
   int i;
   XrdOlbServer *sp;
   SMask_t bmask;

// Obtain a lock on the table and screen out peer nodes
//
   STMutex.Lock();
   bmask = smask & peerMask;

// Run through the table looking for servers to send messages to
//
   for (i = 0; i <= STHi; i++)
       {if ((sp = ServTab[i]) && sp->isServer(bmask) && !sp->isOffline)
           sp->Lock();
           else continue;
        STMutex.UnLock();
        sp->Send(buff, blen);
        sp->UnLock();
        STMutex.Lock();
       }
   STMutex.UnLock();
}

/******************************************************************************/

void XrdOlbManager::Broadcast(SMask_t smask, const struct iovec *iod, int iovcnt)
{
   int i;
   XrdOlbServer *sp;
   SMask_t bmask;

// Obtain a lock on the table and screen out peer nodes
//
   STMutex.Lock();
   bmask = smask & peerMask;

// Run through the table looking for servers to send messages to
//
   for (i = 0; i <= STHi; i++)
       {if ((sp = ServTab[i]) && sp->isServer(bmask) && !sp->isOffline)
           sp->Lock();
           else continue;
        STMutex.UnLock();
        sp->Send(iod, iovcnt);
        sp->UnLock();
        STMutex.Lock();
       }
   STMutex.UnLock();
}

/******************************************************************************/
/*                               g e t M a s k                                */
/******************************************************************************/

SMask_t XrdOlbManager::getMask(unsigned int IPv4adr)
{
   int i;
   XrdOlbServer *sp;
   SMask_t smask = 0;

// Obtain a lock on the table
//
   STMutex.Lock();

// Run through the table looking for server with matching IP address
//
   for (i = 0; i <= STHi; i++)
       if ((sp = ServTab[i]) && sp->isServer(IPv4adr))
          {smask = sp->ServMask; break;}

// All done
//
   STMutex.UnLock();
   return smask;
}
  
/******************************************************************************/
/*                                I n f o r m                                 */
/******************************************************************************/
  
void XrdOlbManager::Inform(const char *cmd, int clen, char *arg, int alen)
{
   EPNAME("Inform");
   int i, iocnt, eol;
   struct iovec iod[4];
   XrdOlbServer *sp;

// Set up i/o vector
//
   iod[0].iov_base = Config.MsgGID;
   iod[0].iov_len  = Config.MsgGIDL;
   iod[1].iov_base = (char *)cmd;
   iod[1].iov_len  = (clen ? clen : strlen(cmd));
   if (!arg) {iocnt = 1; eol = (*(cmd+iod[1].iov_len-1) == '\n');}
      else {iod[2].iov_base = arg;
            iod[2].iov_len  = (alen ? alen : strlen(arg));
            eol = (*(arg+iod[2].iov_len-1) == '\n');
            iocnt = 2;
           }
   if (!eol)
      {iocnt++;
       iod[iocnt].iov_base = (char *)"\n";
       iod[iocnt].iov_len  = 1;
      }
   iocnt++;

// Obtain a lock on the table
//
   MTMutex.Lock();

// Run through the table looking for managers to send messages to
//
   for (i = 0; i <= MTHi; i++)
       {if ((sp=MastTab[i]) && !sp->isOffline) 
           {sp->Lock();
            MTMutex.UnLock();
            DEBUG(sp->Nick() <<" " <<cmd <<(arg ? arg : (char *)""));
            sp->Send(iod, iocnt);
            sp->UnLock();
            MTMutex.Lock();
           }
       }
   MTMutex.UnLock();
}
  
/******************************************************************************/
  
void XrdOlbManager::Inform(SMask_t mmask, const char *cmd, int clen)
{
   EPNAME("Inform");
   int i, iocnt;
   struct iovec iod[4];
   XrdOlbServer *sp;

// Set up i/o vector
//
   iod[0].iov_base = Config.MsgGID;
   iod[0].iov_len  = Config.MsgGIDL;
   iod[1].iov_base = (char *)cmd;
   iod[1].iov_len  = (clen ? clen : strlen(cmd));
   if (*(cmd+iod[1].iov_len-1) == '\n') iocnt = 2;
      else {iod[2].iov_base = (char *)"\n";
            iod[2].iov_len  = 1;
            iocnt = 3;
           }

// Obtain a lock on the table
//
   MTMutex.Lock();

// Run through the table looking for managers to send messages to
//
   for (i = 0; i <= MTHi; i++)
       {if ((sp=MastTab[i]) && sp->isServer(mmask) && !sp->isOffline) 
           {sp->Lock();
            MTMutex.UnLock();
            DEBUG(sp->Nick() <<" " <<cmd);
            sp->Send(iod, iocnt);
            sp->UnLock();
            MTMutex.Lock();
           }
       }
   MTMutex.UnLock();
}

/******************************************************************************/
/*                           L i s t S e r v e r s                            */
/******************************************************************************/
  
XrdOlbSInfo *XrdOlbManager::ListServers(SMask_t mask, int opts)
{
    const char *reason;
    int i, iend, nump, delay, lsall = opts & OLB_LS_ALL;
    XrdOlbServer *sp;
    XrdOlbSInfo  *sipp = 0, *sip;

// If only one wanted, the select appropriately
//
   STMutex.Lock();
   iend = (opts & OLB_LS_BEST ? 0 : STHi);
   for (i = 0; i <= iend; i++)
       {if (opts & OLB_LS_BEST)
            sp = (Config.sched_RR
                 ? SelbyRef( mask, nump, delay, &reason, 0)
                 : SelbyLoad(mask, nump, delay, &reason, 0));
           else if (((sp = ServTab[i]) || (sp = ServBat[i]))
                &&  !lsall && !(sp->ServMask & mask)) sp = 0;
        if (sp)
           {sip = new XrdOlbSInfo((opts & OLB_LS_IPO) ? 0 : sp->Name(), sipp);
            sip->Mask    = sp->ServMask;
            sip->Id      = sp->ServID;
            sip->IPAddr  = sp->IPAddr;
            sip->Port    = sp->Port;
            sip->Load    = sp->myLoad;
            sip->Util    = sp->DiskTotu;
            sip->RefTotA = sp->RefTotA + sp->RefA;
            sip->RefTotR = sp->RefTotR + sp->RefR;
            if (sp->isOffline) sip->Status  = OLB_SERVER_OFFLINE;
               else sip->Status  = 0;
            if (sp->isDisable) sip->Status |= OLB_SERVER_DISABLE;
            if (sp->isNoStage) sip->Status |= OLB_SERVER_NOSTAGE;
            if (sp->isSuspend) sip->Status |= OLB_SERVER_SUSPEND;
            if (sp->isRW     ) sip->Status |= OLB_SERVER_ISRW;
            if (sp->isMan    ) sip->Status |= OLB_SERVER_ISMANGR;
            if (sp->isPeer   ) sip->Status |= OLB_SERVER_ISPEER;
            if (sp->isProxy  ) sip->Status |= OLB_SERVER_ISPROXY;
            sp->UnLock();
            sipp = sip;
           }
       }
   STMutex.UnLock();

// Return result
//
   return sipp;
}
  
/******************************************************************************/
/*                                 L o g i n                                  */
/******************************************************************************/
  
void *XrdOlbManager::Login(XrdNetLink *lnkp)
{
   EPNAME("Login")
   XrdOlbServer *sp;
   const char *mp, *Ptype = "", *Stype = "";
   char *tp, *theSID;
   int   Status = 0, fdsk = 0, numfs = 1, addedp = 0, port = 0, Sport = 0;
   int   rc, servID, servInst, Rslot, isProxy = 0, isPeer = 0, isServ = 0;
   int   udsk;
   SMask_t servset = 0, newmask;

// Handle the login for the server stream.
//
   if (!(tp = lnkp->GetLine())) return Login_Failed("missing login",lnkp);
   DEBUG("from " <<lnkp->Nick() <<": " <<tp);

// The server may only send a proper login request, check for this
//
   if (!(tp = lnkp->GetToken()) || strcmp(tp, "login")
   ||  !(tp = lnkp->GetToken()))
      return Login_Failed("first command not login", lnkp);

// Director Command: login director
//
   if (!strcmp(tp, "director"))
      {sp = new XrdOlbServer(lnkp);
      if (!(Rslot = RTable.Add(sp)))
         Say.Emsg("Manager","Director", lnkp->Nick(),
                                  "login failed; too many directors.");
         else {Say.Emsg("Manager","Director",lnkp->Nick(),"logged in.");
               DEBUG("Director " <<lnkp->Nick() <<" assigned slot " <<Rslot <<'.'<<sp->Instance);
               sp->Info.Rnum = Rslot;
               sp->Process_Director();
               Say.Emsg("Manager","Director",lnkp->Nick(),"logged out.");
              }
       if (Rslot) RTable.Del(sp);
       delete sp;
       return (void *)0;
      }

// Server Command: login {peer | pproxy | proxy | server} [options]
// Options:        [port <dport>] [nostage] [suspend] [+m:<sport>] [=<sid>] [!]

// Responses:      <id> ping
//                 <id> try <hostlist>
// Server Command: addpath <mode> <path>
//                 start <maxkb> [<numfs> [<totkb> | <util>]]
//
// Note: For now we designate proxies as peers to avoid polling them

        if ((isServ =  !strcmp(tp, "server")))
            Stype = "server";
   else if ((isPeer =  !strcmp(tp, "peer")))
           {Ptype = "peer ";       Status |= (OLB_Special | OLB_isPeer);}
   else if ((isProxy = !strcmp(tp, "pproxy")))
           {Ptype = "peer proxy "; Status |= (OLB_Special | OLB_isPeer);}
   else if ((isProxy = !strcmp(tp, "proxy")))
           {Ptype = "proxy ";      Status |= (OLB_Special | OLB_isProxy);}
   else    return Login_Failed("invalid login role", lnkp);

// Disallow subscriptions we are are configured as a solo manager
//
   if (Config.asSolo())
      return Login_Failed("configuration disallows subscribers", lnkp);

// The server may specify a port number
//
   if ((tp = lnkp->GetToken()) && !strcmp("port", tp))
      {if (!(tp = lnkp->GetToken()))
          return Login_Failed("missing start port value", lnkp);
       if (XrdOuca2x::a2i(Say,"start port value",tp,&port,0,65535))
          return Login_Failed("invalid start port value", lnkp);
       tp = lnkp->GetToken();
      }

// The server may specify nostage
//
   if (tp && !strcmp("nostage", tp))
      {Status |= OLB_noStage;
       tp = lnkp->GetToken();
      }

// The server may specify suspend
//
   if (tp && !strcmp("suspend", tp))
      {Status |= OLB_Suspend;
       tp = lnkp->GetToken();
      }

// The server may specify the version (for self-managed servers)
//
   if (tp && *tp == '+')
      {Status |= OLB_Special;
       tp++;
       if (*tp == 'm') 
          {Status |= OLB_isMan; tp++;
           Stype = (isPeer ? "Manager" : "Supervisor");
           if (*tp == ':')
              if (XrdOuca2x::a2i(Say,"subscription port",tp+1,&Sport,0))
                 return Login_Failed("invalid subscription port", lnkp);
           if (!Sport) Sport = Config.PortTCP;
          }
       tp = lnkp->GetToken();
      }

// Server may specify a stable unique identifier relative to hostname
// Note: If additional parms we must copy the sid to a local buffer
//
   if (tp && *tp == '=') {theSID = tp+1; tp = lnkp->GetToken();}
      else theSID = 0;

// The server may specify that it has been trying for a long time
//
   if (tp && *tp == '!')
      Say.Emsg("Manager",lnkp->Nick(),"has not yet found a cluster slot!");

// Make sure that our role is compatible with the incomming role
//
   mp = 0;
   if (Config.asServer())       // We are a supervisor
      {if (Config.asProxy() && (!isProxy || isPeer))
          mp = "configuration only allows proxies";
          else if (!isServ)
          mp = "configuration disallows peers and proxies";
      } else {                  // We are a manager
       if (Config.asProxy() &&   isServ)
          mp = "configuration only allows peers or proxies";
          else if (isProxy)
          mp = "configuration disallows proxies";
      }
   if (mp) return Login_Failed(mp, lnkp);

// Add the server
//
   if (!(sp = AddServer(lnkp, port, Status, Sport, theSID)))
      return Login_Failed(0, lnkp);
   servID = sp->ServID; servInst = sp->Instance;

// Create a human readable role name for this server
//
   {char buff[64];
    sprintf(buff,"%s%s", Ptype, Stype);
    if (sp->Stype) free(sp->Stype);
    sp->Stype = strdup(buff);
   }

// Immediately send a ping request to validate the login (phase 1)
//
   if (Status & OLB_Special) lnkp->Send("1@0 ping\n", 9);

// At this point, the server will send only addpath commands followed by a start
//
   while((tp = lnkp->GetLine()))
        {DEBUG("Received from " <<lnkp->Nick() <<": " <<tp);
         if (!(tp = lnkp->GetToken())) break;
         if (!strcmp(tp, "start")) break;
         if (strcmp(tp, "addpath"))
            return Login_Failed("invalid command sequence", lnkp, sp);
         if (!(newmask = AddPath(sp)))
            return Login_Failed("invalid addpath command", lnkp, sp);
         servset |= newmask;
         addedp= 1;
        }

// At this point if all went well, start the server
//
   if (!tp) return Login_Failed("missing start", lnkp, sp);

// The server may include the max free space, if need be, on the start
//
   if ((tp = lnkp->GetToken())
   && XrdOuca2x::a2i(Say,"start disk free value",tp,&fdsk,0))
      return Login_Failed("invalid start dsk free value", lnkp, sp);
      else {sp->DiskFree = fdsk; sp->DiskNums = 1; sp->DiskTotu = 0;}

// The server may include the number of file systems, on the start
//
   if (tp && (tp = lnkp->GetToken())
   && XrdOuca2x::a2i(Say, "start numfs value",  tp, &numfs, 0))
      return Login_Failed("invalid start numfs value", lnkp, sp);
      else sp->DiskNums = numfs;

// The server may include the total free space in all file systems, on the start
// For new servers the value will be actual disk utilization.
//
   if (tp && (tp = lnkp->GetToken())
   && XrdOuca2x::a2i(Say, "start totkb value",  tp, &udsk, 0))
      return Login_Failed("invalid start util value", lnkp, sp);
      else {if (udsk > 100 && (udsk = 100 - (fdsk/udsk)) > 100) udsk = 100;
            sp->DiskTotu = udsk;
            }

// Check if we have any special paths. If none, then we must set the cache for
// all entries to indicate that the server bounced.
//
   if (!addedp) 
      {XrdOlbPInfo pinfo;
       pinfo.rovec = sp->ServMask;
       if (sp->isPeer) pinfo.ssvec = sp->ServMask;
       servset = Cache.Paths.Insert("/", &pinfo);
       Cache.Bounce(sp->ServMask);
       Say.Emsg("Manager",sp->Stype,lnkp->Nick(),"defaulted r /");
      }

// Finally set the reference counts for intersecting servers to be the same.
// Additionally, indicate cache refresh will be needed because we have a new
// server that may have files the we already reported on.
//
   ResetRef(servset);
   if (Config.asManager()) Reset();

// Process responses from the server.
//
   tp = sp->isSuspend ? (char *)"logged in suspended." : (char *)"logged in.";
   Say.Emsg("Manager", sp->Stype, sp->Nick(), tp);

   sp->isDisable = 0;
   rc = sp->Process_Responses();

   if (rc == -86) mp = "disconnected";
      else mp = sp->isOffline ? "forced out." : "logged out.";
   Say.Emsg("Manager", sp->Stype, sp->Nick(), mp);

// Remove the server from the configuration if it is still in it. We only delete
// this object if the delete was defered as we may still be in the backup config.
//
   sp->Lock();
   if (sp->isBound) {Remove_Server(0,servID,servInst,rc == -86); sp->UnLock();}
      else if (sp->isGone) {sp->UnLock(); delete sp;}
              else sp->UnLock();
   return (void *)0;
}
  
/******************************************************************************/
/*                               M o n P e r f                                */
/******************************************************************************/
  
void *XrdOlbManager::MonPerf()
{
   XrdOlbServer *sp;
   char *reqst;
   int nldval, i;
   int oldval=0, doping = 0;

// Sleep for the indicated amount of time, then maintain load on each server
//
   while(Config.AskPing)
        {Snooze(Config.AskPing);
         if (--doping < 0) doping = Config.AskPerf;
         STMutex.Lock();
         for (i = 0; i <= STHi; i++)
             {if ((sp = ServTab[i]) && sp->isBound) sp->Lock();
                 else continue;
              STMutex.UnLock();
              if (doping || !Config.AskPerf)
                 {reqst = (char *)"1@0 ping\n"; nldval = 0;
                  if ((oldval = sp->pingpong)) sp->pingpong = 0;
                     else sp->pingpong = -1;
                 } else {
                  reqst = (char *)"1@0 usage\n";oldval = 0;
                  if ((nldval = sp->newload)) sp->newload = 0;
                     else sp->newload = -1;
                 }
              if (oldval < 0 || nldval < 0)
                  Remove_Server("not responding", i, sp->Instance);
                  else sp->Send(reqst);
              sp->UnLock();
              STMutex.Lock();
             }
         STMutex.UnLock();
        }
   return (void *)0;
}
  
/******************************************************************************/
/*                               M o n P i n g                                */
/******************************************************************************/
  
void *XrdOlbManager::MonPing()
{
   XrdOlbServer *sp;
   int i;

// Make sure the manager sends at least one request within twice the ping 
// interval plus a little. If we don't get one, then declare the manager dead 
// and re-initialize the manager connection.
//
   do {Snooze(Config.AskPing*2+13);
       MTMutex.Lock();
       for (i = 0; i < MTHi; i++)
           if ((sp = MastTab[i]))
              {sp->Lock();
               if (sp->isActive) sp->isActive = 0;
                  else {Say.Emsg("Manager", "Manager", sp->Link->Nick(),
                                       "appears to be dead.");
                        sp->isOffline = 1;
                        sp->Link->Close(1);
                       }
               sp->UnLock();
              }
       MTMutex.UnLock();
      } while(1);

// Keep the compiler happy
//
   return (void *)0;
}
  
/******************************************************************************/
/*                               M o n R e f s                                */
/******************************************************************************/
  
void *XrdOlbManager::MonRefs()
{
   XrdOlbServer *sp;
   int  i, snooze_interval = 10*60, loopmax, loopcnt = 0;
   int resetA, resetR, resetAR;

// Compute snooze interval
//
   if ((loopmax = Config.RefReset / snooze_interval) <= 1)
      if (!Config.RefReset) loopmax = 0;
         else {loopmax = 1; snooze_interval = Config.RefReset;}

// Sleep for the snooze interval. If a reset was requested then do a selective
// reset unless we reached our snooze maximum and enough selections have gone
// by; in which case, do a global reset.
//
   do {Snooze(snooze_interval);
       loopcnt++;
       STMutex.Lock();
       resetA  = (SelAcnt >= Config.RefTurn);
       resetR  = (SelRcnt >= Config.RefTurn);
       resetAR = (loopmax && loopcnt >= loopmax && (resetA || resetR));
       if (doReset || resetAR)
           {for (i = 0; i <= STHi; i++)
                if ((sp = ServTab[i])
                &&  (resetAR || (doReset && sp->isServer(resetMask))) )
                    {sp->Lock();
                     if (resetA || doReset) {sp->RefTotA += sp->RefA;sp->RefA=0;}
                     if (resetR || doReset) {sp->RefTotR += sp->RefR;sp->RefR=0;}
                     sp->UnLock();
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
/*                                P a n d e r                                 */
/******************************************************************************/
  
void *XrdOlbManager::Pander(char *manager, int mport)
{
   EPNAME("Pander");
   XrdOlbServer   *sp;
   XrdNetLink     *lp;
   int Role = 0, Status;
   int rc, opts = 0, waits = 6, tries = 6, fails = 0, xport = mport;
   int Lvl = 0, mySID = ManTree.Register();
   char manbuff[256], *reason, *manp = manager;
   const int manblen = sizeof(manbuff);

// Compute the Manager's status (this never changes for managers/supervisors)
//
   if (Config.asPeer())
      if (Config.SUPCount)             Role  = OLB_isPeer | OLB_Suspend;
               else                    Role  = OLB_isPeer;
      else if (Config.asManager())     Role  = OLB_isMan  | OLB_Suspend;
   if (Config.asProxy())               Role |= OLB_isProxy;

// Keep connecting to our manager. If XWait is present, wait for it to
// be turned off first; then try to connect.
//
   do {while(XWait)
            {if (!waits--)
                {Say.Emsg("Manager", "Suspend state still active.");
                 waits = 6;
                }
             Snooze(12);
            }
       if (!ManTree.Trying(mySID, Lvl) && Lvl)
          {DEBUG("Restarting at root node " <<manager <<':' <<mport);
           manp = manager; xport = mport; Lvl = 0;
          }
       DEBUG("Trying to connect to lvl " <<Lvl <<' ' <<manp <<':' <<xport);
       if (!(lp = NetTCPs->Connect(manp, xport, opts)))
          {if (tries--) opts = XRDNET_NOEMSG;
              else {tries = 6; opts = 0;}
           if ((Lvl = myMans.Next(xport,manbuff,manblen)))
                  {Snooze(3); manp = manbuff;}
             else {Snooze(6); 
                   if (manp != manager) fails++;
                   manp = manager; xport = mport;
                  }
           continue;
          }
       opts = 0; tries = waits = 6;

       // Obtain a new server object for this server
       //
       sp = new XrdOlbServer(lp);
       Add_Manager(sp);

       // Login this server. If we are a supervisor login possibly suspended
       //
       reason = 0;
       Status = Role|(XWait ? OLB_Suspend : 0)|(XnoStage ? OLB_noStage : 0);
       if (fails >= 6 && manp == manager) {Status |= OLB_Lost; fails = 0;}

       if (!(rc=sp->Login(Port, Status, Lvl+1)))
          if (!ManTree.Connect(mySID, sp)) rc = -86;
             else rc = sp->Process_Requests();
          else if (rc == 1) reason = (char *)"login to manager failed";

       // Just try again. Note that if were successful in fully connecting to
       // the root then we will continue to try the root.
       //
       Remove_Manager(reason, sp);
       ManTree.Disc(mySID);
       delete sp;

       // Cycle on to the next manager if we have one or snooze and try over
       //
       if (rc != -86 && (Lvl = myMans.Next(xport,manbuff,manblen)))
          {manp = manbuff; continue;}
       Snooze(9); Lvl = 0;
       if (manp != manager) fails++;
       manp = manager; xport = mport;
      } while(1);
    return (void *)0;
}

/******************************************************************************/
/*                         R e m o v e _ S e r v e r                          */
/******************************************************************************/

void XrdOlbManager::Remove_Server(const char *reason, 
                                  int sent, int sinst, int immed)
{
   EPNAME("Remove_Server")
   XrdOlbServer *sp;

// Obtain a lock on the servtab
//
   STMutex.Lock();

// Make sure this server is the right one
//
   if (!(sp = ServTab[sent]) || sp->Instance != sinst)
      {STMutex.UnLock();
       DEBUG("Remove server " <<sent <<'.' <<sinst <<" failed.");
       return;
      }

// Do a partial drop at this point
//
   if (sp->Link) {sp->Link->Close(1);}
   sp->isOffline = 1;
   if (sp->isBound) {sp->isBound = 0; ServCnt--;}

// Compute new state of all servers if we are a reporting manager
//
   if (Config.asManager()) OlbState.Calc(-1, sp->isNoStage, sp->isSuspend);

// If this is an immediate drop request, do so now
//
   if (immed || !Config.DRPDelay) {Drop_Server(sent, sinst); return;}

// If a drop job is already scheduled, update the instance field. Otherwise,
// Schedule a server drop at a future time.
//
   sp->DropTime = time(0)+Config.DRPDelay;
   if (sp->DropJob) sp->DropJob->servInst = sinst;
      else sp->DropJob = new XrdOlbDrop(sent, sinst);

// Document removal
//
   if (reason) 
      Say.Emsg("Manager", sp->Nick(), "scheduled for removal;", reason);
      else DEBUG("Will remove " <<sp->Nick() <<" server " <<sent <<'.' <<sinst);
   STMutex.UnLock();
}

/******************************************************************************/
/*                                 R e s e t                                  */
/******************************************************************************/
  
void XrdOlbManager::Reset()
{
   EPNAME("Reset");
   const char *cmd = "0 rst\n";
   const int cmdln = 6;
   XrdOlbServer *sp;
   int i;

// Obtain a lock on the table
//
   MTMutex.Lock();

// Run through the table looking for managers to send a reset request
//
   for (i = 0; i <= MTHi; i++)
       {if ((sp=MastTab[i]) && !sp->isOffline && sp->isKnown)
           {sp->Lock();
            sp->isKnown = 0;
            MTMutex.UnLock();
            DEBUG("sent to " <<sp->Nick());
            sp->Send(cmd, cmdln);
            sp->UnLock();
            MTMutex.Lock();
           }
       }
   MTMutex.UnLock();
}

/******************************************************************************/
/*                                R e s u m e                                 */
/******************************************************************************/

void XrdOlbManager::Resume()
{
     const char *cmd = "resume\n";
     const int   cln = strlen(cmd);

// If the suspend file is still present, ignore this resume request
//
   if (Config.inSuspend())
      Say.Emsg("Manager","Resume request ignored; suspend file present.");
      else {XWait = 0;
            Inform(cmd, (int)cln);
           }
}
  
/******************************************************************************/
/*                             S e l S e r v e r                              */
/******************************************************************************/
  
int XrdOlbManager::SelServer(int opts, char *path,
                            SMask_t pmask, SMask_t amask, char *hbuff,
                            const struct iovec *iodata, int iovcnt)
{
    EPNAME("SelServer")
    const char *reason, *reason2;
    int delay = 0, delay2 = 0, nump, isalt = 0, pass = 2;
    int needrw = opts & (OLB_needrw | OLB_newfile);
    SMask_t mask;
    XrdOlbServer *sp = 0;

// Scan for a primary and alternate server (alternates do staging). At this
// point we omit all peer servers as they are our last resort.
//
   STMutex.Lock();
   mask = pmask & peerMask;
   while(pass--)
        {if (mask)
            {sp = (Config.sched_RR
                   ? SelbyRef( mask, nump, delay, &reason, isalt||needrw)
                   : SelbyLoad(mask, nump, delay, &reason, isalt||needrw));
             if (sp || (nump && delay) || ServCnt < Config.SUPCount) break;
            }
         mask = amask & peerMask; isalt = 1;
        }
   STMutex.UnLock();

// Update info
//
   if (sp)
      {strcpy(hbuff, sp->Name());
       if (isalt || (opts & OLB_newfile) || iovcnt)
          {if (isalt) Cache.AddFile(path, sp->ServMask, needrw);
           if (iovcnt && iodata) sp->Link->Send(iodata, iovcnt);
                  TRACE(Stage, "Server " <<hbuff <<" staging "  <<path);
          } else {TRACE(Stage, "Server " <<hbuff <<" creating " <<path);}
       sp->UnLock();
       return 0;
      } else if (!delay && ServCnt < Config.SUPCount)
                {reason = "insufficient number of servers";
                 delay = Config.SUPDelay;
                }

// Return delay if selection failure is recoverable
//
   if (delay && delay < Config.PSDelay)
      {Record(path, reason);
       return delay;
      }

// At this point, we attempt a peer node selection (choice of last resort)
//
   if (opts & OLB_peersok)
      {STMutex.Lock();
       if ((mask = (pmask | amask) & peerHost))
          sp = SelbyCost(mask, nump, delay2, &reason2, needrw);
       STMutex.UnLock();
       if (sp)
          {strcpy(hbuff, sp->Name());
           if (iovcnt && iodata) sp->Link->Send(iodata, iovcnt);
           sp->UnLock();
           TRACE(Stage, "Peer " <<hbuff <<" handling " <<path);
           return 0;
          }
       if (!delay) {delay = delay2; reason = reason2;}
      }

// At this point we either don't have enough servers or simply can't handle this
//
   if (delay)
      {TRACE(Defer, "client defered; " <<reason <<" for " <<path);
       return delay;
      }
   return -1;
}

/******************************************************************************/
  
int XrdOlbManager::SelServer(int isrw, SMask_t pmask, char *hbuff)
{
   static const SMask_t smLow = 255;
   XrdOlbServer *sp = 0;
   SMask_t tmask;
   int Snum = 0;

// Compute the a single server number that is contained in the mask
//
   if (!pmask) return 0;
   do {if (!(tmask = pmask & smLow)) Snum += 8;
         else {while((tmask = tmask>>1)) Snum++; break;}
      } while((pmask = pmask >> 8));

// See if the server passes muster
//
   STMutex.Lock();
   if ((sp = ServTab[Snum]))
      {if (sp->isOffline || sp->isSuspend || sp->isDisable)            sp = 0;
          else if (!Config.sched_RR
               && (sp->myLoad > Config.MaxLoad))                 sp = 0;
       if (sp)
          if (isrw)
             if (sp->isNoStage || sp->DiskFree < Config.DiskMin) sp = 0;
                else {SelAcnt++; sp->Lock();}
            else     {SelRcnt++; sp->Lock();}
      }
   STMutex.UnLock();

// At this point either we have a server or we do not
//
   if (sp)
      {strcpy(hbuff, sp->Name());
       sp->RefR++;
       sp->UnLock();
       return 1;
      }
   return 0;
}

/******************************************************************************/
/*                              R e s e t R e f                               */
/******************************************************************************/
  
void XrdOlbManager::ResetRef(SMask_t smask)
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

/********************************************************************************/
/*                                S n o o z e                                 */
/******************************************************************************/
  
void XrdOlbManager::Snooze(int slpsec)
{
   int retc;
   struct timespec lftp, rqtp = {slpsec, 0};

   while ((retc = nanosleep(&rqtp, &lftp)) < 0 && errno == EINTR)
         {rqtp.tv_sec  = lftp.tv_sec;
          rqtp.tv_nsec = lftp.tv_nsec;
         }

   if (retc < 0) Say.Emsg("Manager", errno, "sleep");
}

/******************************************************************************/
/*                                 S p a c e                                  */
/******************************************************************************/
  
void XrdOlbManager::Space(int none, int doinform)
{
     const char *cmd = (none ? "nostage" : "stage");
     int PStage;

     if (Config.asSolo()) return;

     XXMutex.Lock();
     PStage = XnoStage;
     if (none) {XnoStage |=  OLB_SERVER_NOSPACE; PStage = !PStage;}
        else    XnoStage &= ~OLB_SERVER_NOSPACE;
     if (doinform && PStage) Inform(cmd, strlen(cmd));
     XXMutex.UnLock();
}

/******************************************************************************/
/*                                 S t a g e                                  */
/******************************************************************************/
  
void XrdOlbManager::Stage(int ison, int doinform)
{
     const char *cmd = (ison ? "stage" : "nostage");
     int PStage;

     XXMutex.Lock();
     PStage = XnoStage;
     if (ison)  XnoStage &= ~OLB_SERVER_NOSTAGE;
        else   {XnoStage |=  OLB_SERVER_NOSTAGE; PStage = !PStage;}
     if (doinform && PStage) Inform(cmd, strlen(cmd));
     XXMutex.UnLock();
}

/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/
  
int XrdOlbManager::Stats(char *bfr, int bln)
{
   static const char statfmt1[] = "<stats id=\"olb\"><name>%s</name>";
   static const char statfmt2[] = "<subscriber><name>%s</name>"
          "<status>%s</status><load>%d</load><diskfree>%d</diskfree>"
          "<refa>%d</refa><refr>%d</refr></subscriber>";
   static const char statfmt3[] = "</stats>\n";
   XrdOlbSInfo *sp;
   int mlen, tlen = sizeof(statfmt3);
   char stat[6], *stp;

   class spmngr {
         public: XrdOlbSInfo *sp;

                 spmngr() {sp = 0;}
                ~spmngr() {XrdOlbSInfo *xsp;
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
   mngrsp.sp = sp = ListServers();

// Format the statistics
//
   mlen = snprintf(bfr, bln, statfmt1, Config.myName);
   if ((bln -= mlen) <= 0) return 0;
   tlen += mlen;

   while(sp && bln)
        {stp = stat;
         if (sp->Status)
            {if (sp->Status & OLB_SERVER_OFFLINE) *stp++ = 'o';
             if (sp->Status & OLB_SERVER_SUSPEND) *stp++ = 's';
             if (sp->Status & OLB_SERVER_NOSTAGE) *stp++ = 'n';
             if (sp->Status & OLB_SERVER_DISABLE) *stp++ = 'd';
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
/*                               S u s p e n d                                */
/******************************************************************************/

void XrdOlbManager::Suspend(int doinform)
{
     const char *cmd = "suspend\n";
     const int   cln = strlen(cmd);

     XWait = 1;
     if (doinform) Inform(cmd, (int)cln);
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               A d d P a t h                                */
/******************************************************************************/
  
SMask_t XrdOlbManager::AddPath(XrdOlbServer *sp)
{
    char *tp;
    XrdOlbPInfo pinfo;

// Process: addpath {r | w | rw}[s] path
//
   if (!(tp = sp->Link->GetToken())) return 0;
   while(*tp)
        {     if ('r' == *tp) pinfo.rovec =               sp->ServMask;
         else if ('w' == *tp) pinfo.rovec = pinfo.rwvec = sp->ServMask;
         else if ('s' == *tp) pinfo.rovec = pinfo.ssvec = sp->ServMask;
         else return 0;
         tp++;
        }

// Get the path
//
   if (!(tp = sp->Link->GetToken())) return 0;
   if (pinfo.rwvec || pinfo.ssvec) sp->isRW = 1;

// For everything matching the path indicate in the cache the server bounced
//
   Cache.Bounce(sp->ServMask, tp);

// Add the path to the known path list
//
   return Cache.Paths.Insert(tp, &pinfo);
}

/******************************************************************************/
/*                           A d d _ M a n a g e r                            */
/******************************************************************************/
  
int XrdOlbManager::Add_Manager(XrdOlbServer *sp)
{
   EPNAME("AddManager")
   const SMask_t smask_1 = 1;
   int i;

// Find available ID for this server
//
   MTMutex.Lock();
   for (i = 0; i < MTMax; i++) if (!MastTab[i]) break;

// Check if we have too many here
//
   if (i > MTMax)
      {MTMutex.UnLock();
       Say.Emsg("Manager", "Login to", sp->Link->Nick(),
                     "failed; too many managers");
       return 0;
      }

// Assign new manager
//
   MastTab[i] = sp;
   if (i > MTHi) MTHi = i;
   sp->ServID   = i;
   sp->ServMask = smask_1<<i;
   sp->isOffline  = 0;
   sp->isNoStage  = 0;
   sp->isSuspend  = 0;
   sp->isActive   = 1;
   sp->isMan      = (Config.asManager() ? 1 : 0);
   MTMutex.UnLock();

// Document login
//
   DEBUG("Added " <<sp->Nick() <<" to manager config; id=" <<i);
   return 1;
}

/******************************************************************************/
/*                             A d d S e r v e r                              */
/******************************************************************************/
  
XrdOlbServer *XrdOlbManager::AddServer(XrdNetLink *lp, int port, 
                                           int Status, int sport, char *theSID)
{
   EPNAME("AddServer")
    const SMask_t smask_1 = 1;
    int tmp, i, j = -1, k = -1, os = -1;
    const char *act = "Added ";
    const char *hnp = lp->Nick();
    unsigned int ipaddr = lp->Addr();
    XrdOlbServer *bump = 0, *sp = 0;

// Find available ID for this server
//
   STMutex.Lock();
   for (i = 0; i < STMax; i++)
       if (ServBat[i])
          {if (ServBat[i]->isServer(ipaddr, port, theSID)) break;
              else if (!ServTab[i] && k < 0) k = i;
                   else if (os < 0 && (Status & (OLB_isMan|OLB_isPeer))
                        &&  ServBat[i]->isSpecial) os = i;
          }
          else if (j < 0) j = i;

// Check if server is already logged in or is a relogin
//
   if (i < STMax)
      if (ServTab[i] && ServTab[i]->isBound)
         {STMutex.UnLock();
          Say.Emsg("Manager", "Server", hnp, "already logged in.");
          return 0;
         } else { // Rehook server to previous entry
          sp = ServBat[i];
          if (sp->Link) sp->Link->Recycle();
          sp->Link      = lp;
          sp->isOffline = 0;
          sp->isBusy    = 1;
          sp->Instance++;
          sp->setName(lp, port);  // Just in case it changed
          ServTab[i] = sp;
          j = i;
          act = "Re-added ";
         }

// Reuse an old ID if we must
//
   if (!sp)
      {if (j < 0 && k >= 0)
          {DEBUG("ID " <<k <<" reassigned" <<ServBat[k]->Nick() <<" to " <<hnp);
           if (ServBat[k]->isBusy) ServBat[k]->isGone = 1;
              else delete ServBat[k]; 
           ServBat[k] = 0; j = k;
          } else if (j < 0)
                    {if (os >= 0)
                        {bump = ServTab[os];
                         ServTab[os] = ServBat[os] = 0;
                         j = os;
                        } else {STMutex.UnLock();
                                if (Status&OLB_Special && !(Status&OLB_isPeer))
                                   {sendAList(lp);
                                    act = " redirected";
                                   } else act = " failed";
                                DEBUG("Login " <<hnp <<act 
                                      <<"; too many subscribers");
                                return 0;
                               }
                    }
       ServTab[j]   = ServBat[j] = sp = new XrdOlbServer(lp, port, theSID);
       sp->ServID   = j;
       sp->ServMask = smask_1<<j;
      }

// Check if we should bump someone now as we cannot go through Remove_Server().
//
   if (bump)
      {bump->Lock();
       sendAList(bump->Link);
       DEBUG("ID " <<os <<" bumped " <<bump->Nick() <<" for " <<hnp);
       bump->isOffline = 1;
       bump->isGone    = 1;
       bump->Link->Close(1);
       ServCnt--;
       bump->UnLock();
      }

// Indicate whether this server can be redirected
//
   if (Status & (OLB_isMan | OLB_isPeer)) sp->isSpecial = 0;
      else sp->isSpecial = (Status & OLB_Special);

// Assign new server
//
   if (Status & OLB_isMan) setAltMan(j, ipaddr, sport);
   if (j > STHi) STHi = j;
   sp->isBound   = 1;
   sp->isBusy    = 1;
   sp->isNoStage = (Status & OLB_noStage);
   sp->isSuspend = (Status & OLB_Suspend);
   sp->isMan     = (Status & OLB_isMan);
   sp->isPeer    = (Status & OLB_isPeer);
   sp->isDisable = 1;
   ServCnt++;
   if (Config.SUPLevel
   && (tmp = ServCnt*Config.SUPLevel/100) > Config.SUPCount)
      Config.SUPCount=tmp;

// Compute new peer mask, as needed
//
   if (sp->isPeer) peerHost |=  sp->ServMask;
      else         peerHost &= ~sp->ServMask;
   peerMask = ~peerHost;

// Document login
//
   DEBUG(act <<sp->Nick() <<" to server config; id=" <<j <<'.' <<sp->Instance
         <<"; num=" <<ServCnt <<"; min=" <<Config.SUPCount);

// Compute new state of all servers if we are a reporting manager.
//
   if (Config.asManager()) OlbState.Calc(1, sp->isNoStage, sp->isSuspend);
   STMutex.UnLock();

// All done, all locks released, return the server
//
   return sp;
}

/******************************************************************************/
/*                             c a l c D e l a y                              */
/******************************************************************************/
  
XrdOlbServer *XrdOlbManager::calcDelay(int nump, int numd, int numf, int numo,
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
   return (XrdOlbServer *)0;
}

/******************************************************************************/
/*                           D r o p _ S e r v e r                            */
/******************************************************************************/
  
// Warning: STMutex must be locked upon entry. It will be released upon exit!
//          This method may only be called via Remove_Server() either directly
//          or via a defered job scheduled by that method.

int XrdOlbManager::Drop_Server(int sent, int sinst, XrdOlbDrop *djp)
{
   EPNAME("Drop_Server")
   XrdOlbServer *sp;
   char hname[256];

// Make sure this server is the right one
//
   if (!(sp = ServTab[sent]) || sp->Instance != sinst)
      {if (djp == sp->DropJob) {sp->DropJob = 0; sp->DropTime = 0;}
       DEBUG("Drop server " <<sent <<'.' <<sinst <<" cancelled.");
       STMutex.UnLock();
       return 0;
      }

// Check if the drop has been rescheduled
//
   if (djp && time(0) < sp->DropTime)
      {Sched->Schedule((XrdJob *)djp, sp->DropTime);
       STMutex.UnLock();
       return 1;
      }

// Save the server name (don't want to hold a lock across a message)
//
   strncpy(hname, sp->Nick(), sizeof(hname)-1);
   hname[sizeof(hname)-1] = '\0';

// Remove server from the manager table
//
   ServTab[sent] = 0;
   sp->isOffline = 1;
   sp->DropTime  = 0;
   sp->DropJob   = 0;
   sp->isBound   = 0;

// Remove Server from the peer list (if it is one)
//
   if (sp->isPeer) {peerHost &= sp->ServMask; peerMask = ~peerHost;}

// Remove server entry from the alternate list and readjust the end pointer.
//
   if (sp->isMan)
      {memset((void *)&AltMans[sent*AltSize], (int)' ', AltSize);
       if (sent == AltMent)
          {AltMent--;
           while(AltMent >= 0 &&  ServTab[AltMent]
                              && !ServTab[AltMent]->isMan) AltMent--;
           if (AltMent < 0) AltMend = AltMans;
              else AltMend = AltMans + ((AltMent+1)*AltSize);
          }
      }

// Readjust STHi
//
   if (sent == STHi) while(STHi >= 0 && !ServTab[STHi]) STHi--;

// Document the drop (The link will be recycled when object is actually deleted)
//
   STMutex.UnLock();
   DEBUG("Server " <<hname <<' ' <<sent <<'.' <<sinst <<" dropped.");
   Say.Emsg("Drop_Server", hname, "dropped.");
   return 0;
}

/******************************************************************************/
/*                          L o g i n _ F a i l e d                           */
/******************************************************************************/
  
void *XrdOlbManager::Login_Failed(const char *reason, 
                                 XrdNetLink *lp, XrdOlbServer *sp)
{
// If we have a server object then we must remove it and then delete it. This
// is safe to do because we ask for an immediate drop. The delete will recycle
// the link so we need not do it here. Otherwise, a link pointer must be
// passed which we will recycle as a server object does not yet exist.
//
     if (sp) {Remove_Server(reason, sp->ServID, sp->Instance, 1);
              delete sp;
             }
        else {if (reason) Say.Emsg("Manager", lp->Nick(),
                                          "login failed;", reason);
              lp->Recycle();
             }
     return (void *)0;
}

/******************************************************************************/
/*                                R e c o r d                                 */
/******************************************************************************/
  
void XrdOlbManager::Record(char *path, const char *reason)
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
/*                        R e m o v e _ M a n a g e r                         */
/******************************************************************************/

void XrdOlbManager::Remove_Manager(const char *reason, XrdOlbServer *sp)
{
   EPNAME("Remove_Manager")
   int sent  = sp->ServID;
#ifndef NODEBUG
   int sinst = sp->Instance;
#endif

// Obtain a lock on the servtab
//
   MTMutex.Lock();

// Make sure this server is the right one
//
   if (!(sp == MastTab[sent]))
      {MTMutex.UnLock();
       DEBUG("Remove manager " <<sent <<'.' <<sinst <<" failed.");
       return;
      }

// Remove server from the manager table
//
   MastTab[sent] = 0;
   sp->isOffline = 1;
   DEBUG("Removed " <<sp->Nick() <<" manager " <<sent <<'.' <<sinst <<" FD=" <<sp->Link->FDnum());

// Readjust MTHi
//
   if (sent == MTHi) while(MTHi >= 0 && !MastTab[MTHi]) MTHi--;
   MTMutex.UnLock();

// Document removal
//
   if (reason) Say.Emsg("Manager", sp->Nick(), "removed;", reason);
}
 
/******************************************************************************/
/*                             S e l b y C o s t                              */
/******************************************************************************/

// Cost selection is used only for peer node selection as peers do not
// report a load and handle their own scheduling.

XrdOlbServer *XrdOlbManager::SelbyCost(SMask_t mask, int &nump, int &delay,
                                       const char **reason, int needspace)
{
    int i, numd, numf, nums;
    XrdOlbServer *np, *sp = 0;

// Scan for a server
//
   nump = nums = numf = numd = 0; // possible, suspended, full, and dead
   for (i = 0; i <= STHi; i++)
       if ((np = ServTab[i]) && (np->ServMask & mask))
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

// Check for overloaded server and return result
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
  
XrdOlbServer *XrdOlbManager::SelbyLoad(SMask_t mask, int &nump, int &delay,
                                       const char **reason, int needspace)
{
    int i, numd, numf, numo, nums;
    XrdOlbServer *np, *sp = 0;

// Scan for a server (preset possible, suspended, overloaded, full, and dead)
//
   nump = nums = numo = numf = numd = 0; 
   for (i = 0; i <= STHi; i++)
       if ((np = ServTab[i]) && (np->ServMask & mask))
          {nump++;
           if (np->isOffline)                     {numd++; continue;}
           if (np->isSuspend || np->isDisable)    {nums++; continue;}
           if (np->myLoad > Config.MaxLoad) {numo++; continue;}
           if (needspace && (   np->isNoStage
                             || np->DiskFree < Config.DiskMin))
              {numf++; continue;}
           if (!sp) sp = np;
              else if (abs(sp->myLoad - np->myLoad)
                          <= Config.P_fuzz)
                      {if (needspace)
                          {if (sp->RefA > (np->RefA+Config.DiskLinger))
                               sp=np;
                           } 
                           else if (sp->RefR > np->RefR) sp=np;
                       }
                       else if (sp->myLoad > np->myLoad) sp=np;
          }

// Check for overloaded server and return result
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

XrdOlbServer *XrdOlbManager::SelbyRef(SMask_t mask, int &nump, int &delay,
                                      const char **reason, int needspace)
{
    int i, numd, numf, nums;
    XrdOlbServer *np, *sp = 0;

// Scan for a server
//
   nump = nums = numf = numd = 0; // possible, suspended, full, and dead
   for (i = 0; i <= STHi; i++)
       if ((np = ServTab[i]) && (np->ServMask & mask))
          {nump++;
           if (np->isOffline)                   {numd++; continue;}
           if (np->isSuspend || np->isDisable)  {nums++; continue;}
           if (needspace && (   np->isNoStage
                             || np->DiskFree < Config.DiskMin))
              {numf++; continue;}
           if (!sp) sp = np;
              else if (needspace)
                      {if (sp->RefA > (np->RefA+Config.DiskLinger)) sp=np;}
                      else if (sp->RefR > np->RefR) sp=np;
          }

// Check for overloaded server and return result
//
   if (!sp) return calcDelay(nump, numd, numf, 0, nums, delay, reason);
   sp->Lock();
   if (needspace) {SelAcnt++; sp->RefA++;}  // Protected by STMutex
      else        {SelRcnt++; sp->RefR++;}
   delay = 0;
   return sp;
}
 
/******************************************************************************/
/*                             s e n d A L i s t                              */
/******************************************************************************/
  
// Single entry at a time, protected by STMutex!

void XrdOlbManager::sendAList(XrdNetLink *lp)
{
   static char *AltNext = AltMans;
   static struct iovec iov[4] = {{(caddr_t)"0 try ", 6}, 
                                 {0, 0},
                                 {AltMans, 0},
                                 {(caddr_t)"\n", 1}};
// Calculate what to send
//
   AltNext = AltNext + AltSize;
   if (AltNext >= AltMend)
      {AltNext = AltMans;
       iov[1].iov_len = 0;
       iov[2].iov_len = AltMend - AltMans;
      } else {
        iov[1].iov_base = (caddr_t)AltNext;
        iov[1].iov_len  = AltMend - AltNext;
        iov[2].iov_len  = AltNext - AltMans;
      }

// Send the list of alternates (rotated once)
//
   lp->Send(iov, 4);
}

/******************************************************************************/
/*                             s e t A l t M a n                              */
/******************************************************************************/
  
// Single entry at a time, protected by STMutex!
  
void XrdOlbManager::setAltMan(int snum, unsigned int ipaddr, int port)
{
   char *ap = &AltMans[snum*AltSize];
   int i;

// Preset the buffer and pre-screen the port number
//
   if (!port || (port > 0x0000ffff)) port = Port;
   memset(ap, int(' '), AltSize);

// Insert the ip address of this server into the list of servers
//
   i = XrdNetDNS::IP2String(ipaddr, port, ap, AltSize);
   ap[i] = ' ';

// Compute new fence
//
   if (ap >= AltMend) {AltMend = ap + AltSize; AltMent = snum;}
}
