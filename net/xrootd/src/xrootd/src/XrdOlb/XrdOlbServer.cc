/******************************************************************************/
/*                                                                            */
/*                       X r d O l b S e r v e r . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOlbServerCVSID = "$Id$";
  
#include <limits.h>
#include <stdio.h>
#include <utime.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Xrd/XrdJob.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdNet/XrdNetLink.hh"
#include "XrdOlb/XrdOlbCache.hh"
#include "XrdOlb/XrdOlbConfig.hh"
#include "XrdOlb/XrdOlbManager.hh"
#include "XrdOlb/XrdOlbManList.hh"
#include "XrdOlb/XrdOlbMeter.hh"
#include "XrdOlb/XrdOlbPrepare.hh"
#include "XrdOlb/XrdOlbRRQ.hh"
#include "XrdOlb/XrdOlbServer.hh"
#include "XrdOlb/XrdOlbState.hh"
#include "XrdOlb/XrdOlbTrace.hh"
#include "XrdOlb/XrdOlbXmi.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdOlb;

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

       XrdNetLink    *XrdOlbServer::Relay = 0;

/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/

XrdSysMutex XrdOlbServer::mlMutex;

int         XrdOlbServer::xeq_load = 0;
int         XrdOlbServer::cpu_load = 0;
int         XrdOlbServer::mem_load = 0;
int         XrdOlbServer::pag_load = 0;
int         XrdOlbServer::net_load = 0;
int         XrdOlbServer::dsk_free = 0;
int         XrdOlbServer::dsk_totu = 0;
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOlbServer::XrdOlbServer(XrdNetLink *lnkp, int port, char *sid)
                          : Req(this, &Info)
{
    static XrdSysMutex iMutex;
    static int         iNum = 1;

    Link     =  lnkp;
    IPAddr   =  (lnkp ? lnkp->Addr() : 0);
    ServMask =  0;
    ServID   = -1;
    isDisable=  0;
    isNoStage=  0;
    isOffline=  (lnkp == 0);
    isSuspend=  0;
    isActive =  0;
    isBound  =  0;
    isBusy   =  0;
    isGone   =  0;
    isSpecial=  0;
    isMan    =  0;
    isKnown  =  0;
    isPeer   =  0;
    isProxy  =  0;
    myCost   =  0;
    myLoad   =  0;
    DiskFree =  0;
    DiskNums =  0;
    DiskTotu =  0;
    newload  =  1;
    Next     =  0;
    RefA     =  0;
    RefTotA  =  0;
    RefR     =  0;
    RefTotR  =  0;
    pingpong =  0;
    logload  =  Config.LogPerf;
    DropTime =  0;
    DropJob  =  0;
    myName   =  0;
    myNick   =  0;
    Port     =  0; // setName() will set myName, myNick, and Port!
    setName(lnkp, port);
    Stype    =  0;
    mySID    = strdup(sid ? sid : "?");
    myLevel  = 0;

    iMutex.Lock();
    Instance =  iNum++;
    iMutex.UnLock();
    Info.Rinst = Instance;

    redr_iov[1].iov_base = (char *)" !try ";
    redr_iov[1].iov_len  = 6;
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdOlbServer::~XrdOlbServer()
{
// Lock server
//
   Lock(); 

// If we are still an attached server, remove ourselves
//
   if (ServMask)
      {Cache.Paths.Remove(ServMask);
       Cache.Reset(ServID);
      }
   isOffline = 1;

// Recycle the link
//
   if (Link) Link->Recycle(); 
   Link = 0;

// Delete other appendages
//
   if (myName)    free(myName);
   if (myNick)    free(myNick);
   if (mySID)     free(mySID);
   if (Stype)     free(Stype);

// All done
//
   UnLock();
}

/******************************************************************************/
/*                                 L o g i n                                  */
/******************************************************************************/
  
int XrdOlbServer::Login(int dataPort, int Status, int Lvl)
{
   XrdOlbPList *plp = Config.PathList.First();
   int  tutil, maxfr;
   char pbuff[16], qbuff[16], buff[1280];
   const char *role;

// Send a login request
//
   myLevel = Lvl;
   if (dataPort) sprintf(pbuff, "port %d", dataPort);
      else *pbuff = '\0';
   if (Status & OLB_isPeer) *qbuff = '\0';
      else sprintf(qbuff,"+%c:%d",(Status&OLB_isMan?'m':'s'),Config.PortTCP);

   if (Status & OLB_isProxy) role = (Status & OLB_isPeer ? "pproxy" : "proxy");
      else                   role = (Status & OLB_isPeer ? "peer"   : "server");

   sprintf(buff, "login %s %s %s %s %s =%s%s\n", role, pbuff,
                 (Status & OLB_noStage ? "nostage" : ""),
                 (Status & OLB_Suspend ? "suspend" : ""),
                 qbuff, Config.mySID, (Status & OLB_Lost ? " !" : ""));
   if (Link->Send(buff) < 0) return -1;

// If this is a new manager, it will send us a ping or try response at this
// point. Try to receive the response but wait no more than 5 seconds.
//
   if (Link->OK2Recv(5000))
      {char *tp, id[16];
       if ((tp = Receive(id, sizeof(id))))
          if (!strcmp(tp, "try")) 
             {unsigned int ipaddr = Link->Addr();
              myMans.Del(ipaddr);
              while((tp = Link->GetToken()))
                   myMans.Add(ipaddr, tp, Config.PortTCP, Lvl);
              return 2;
             }
      }

// Send all of the addpath commands we need to
//
   while(plp)
        {if (Link->Send(buff, snprintf(buff, sizeof(buff)-1,
                        "addpath %s %s\n", plp->PType(), plp->Path())) < 0)
            return -1;
         plp = plp->Next();
        }

// Now issue a start. For supervisors use zero-knowledge info. A usage
// report will be automatically sent the moment we get some free space.
//
   if (Config.asManager())
      {if (Link->Send("start 0 1 0\n") < 0) return -1;
      } else {
       maxfr = Meter.FreeSpace(tutil);
       if (Link->Send(buff, snprintf(buff, sizeof(buff)-1,
                      "start %d %d %d\n",
                      maxfr, Meter.numFS(), tutil)) < 0) return -1;
      }

// Document the login
//
   Say.Emsg("Server", "Logged into", Link->Name());
   return 0;
}

/******************************************************************************/
/*                      P r o c e s s _ D i r e c t o r                       */
/******************************************************************************/
  
void XrdOlbServer::Process_Director()
{
   char *tp, id[16];
   int retc;

// Simply keep reading commands from the director and process them
// The command sequence is <id> <command> <args>
//
   do {     if (!(tp = Receive(id, sizeof(id)))) retc = -1;
       else if (Config.Disabled)       retc = do_Delay(id);
       else if (!strcmp("select", tp)) retc = do_Select(id);
       else if (!strcmp("selects",tp)) retc = do_Select(id, 1);
       else if (!strcmp("prepadd",tp)) retc = do_PrepAdd(id);
       else if (!strcmp("prepdel",tp)) retc = do_PrepDel(id);
       else if (!strcmp("statsz", tp)) retc = do_Stats(id,0);
       else if (!strcmp("stats",  tp)) retc = do_Stats(id,1);
       else if (!strcmp("chmod",  tp)) retc = do_Chmod(id, 0);
       else if (!strcmp("mkdir",  tp)) retc = do_Mkdir(id, 0);
       else if (!strcmp("mkpath", tp)) retc = do_Mkpath(id, 0);
       else if (!strcmp("mv",     tp)) retc = do_Mv(id, 0);
       else if (!strcmp("rm",     tp)) retc = do_Rm(id, 0);
       else if (!strcmp("rmdir",  tp)) retc = do_Rmdir(id, 0);
       else retc = 1;
       if (retc > 0)
          Say.Emsg("Director", "invalid request from", Link->Name());
      } while(retc >= 0);

// If we got here, then the server connection was closed
//
   if (!isOffline)
     {isOffline = 1;
      if ((retc = Link->LastError()))
         Say.Emsg("Server", retc, "read request from", Link->Name());
     }
}

/******************************************************************************/
/*                      P r o c e s s _ R e q u e s t s                       */
/******************************************************************************/

// Process_Requests handles manager requests on the server's side. These
// requests will be forwarded if we have any subscribers.
//
int XrdOlbServer::Process_Requests()
{
   char *tp, id[16];
   int retc;

// If we are a line manager then we must synchronize our state with our manager
//
   if (Config.asManager()) OlbState.Sync(ServMask,1,1);

// Simply keep reading all requests until the link closes
//
   do {     if (!(tp = Receive(id, sizeof(id)))) retc = -1;
       else if (!strcmp("state",  tp)) retc = do_State(id, 0);
       else if (!strcmp("statf",  tp)) retc = do_State(id, 1);
       else if (!strcmp("ping",   tp)) retc = do_Ping(id);
       else if (!strcmp("prepadd",tp)) retc = do_PrepAdd(id,Manager.hasData);
       else if (!strcmp("prepdel",tp)) retc = do_PrepDel(id,Manager.hasData);
       else if (!strcmp("usage",  tp)) retc = do_Usage(id);
       else if (!strcmp("space",  tp)) retc = do_Space(id);
       else if (!strcmp("chmod",  tp)) retc = do_Chmod(id,  Manager.hasData);
       else if (!strcmp("mkdir",  tp)) retc = do_Mkdir(id,  Manager.hasData);
       else if (!strcmp("mkpath", tp)) retc = do_Mkpath(id, Manager.hasData);
       else if (!strcmp("mv",     tp)) retc = do_Mv(id,     Manager.hasData);
       else if (!strcmp("rm",     tp)) retc = do_Rm(id,     Manager.hasData);
       else if (!strcmp("rmdir",  tp)) retc = do_Rmdir(id,  Manager.hasData);
       else if (!strcmp("try",    tp)) retc = do_Try(id);
       else if (!strcmp("disc",   tp)) retc = do_Disc(id, 0);
       else retc = 1;
       if (retc > 0)
          Say.Emsg("Server", "invalid request from", Link->Name());
      } while(retc >= 0 && !isOffline);

// Check for permanent errors
//
   if (!isOffline)
      {isOffline = 1;
       if (retc < 0 && Link->LastError())
          Say.Emsg("Server", Link->LastError(), "read response from",
                         Link->Name());
      }

// All done
//
   return retc;
}
  
/******************************************************************************/
/*                     P r o c e s s _ R e s p o n s e s                      */
/******************************************************************************/
  
// Process_Responses handles server responses on the manager's side. This is
// a manager function and responses will be propogated up if we are subscribed.
//
int XrdOlbServer::Process_Responses()
{
   char *tp, id[16];
   int retc;

// For newly logged in servers, we need to sync the free space stats
//
   if (Config.asManager())
      {mlMutex.Lock();
       if (isRW && DiskFree > dsk_free)
          {retc = dsk_free; dsk_free = DiskFree; dsk_totu = DiskTotu;
           if (!retc)
              {char respbuff[128];
               Manager.Inform("avkb", 4,
                               respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                               " %d %d\n", dsk_free, dsk_totu));
              }
          }
       mlMutex.UnLock();
      }

// Read all of the server's responses until eof then return
//
   do {     if (!(tp = Receive(id, sizeof(id)))) retc = -1;
       else if (!strcmp("have",    tp)) retc = do_Have(id);
       else if (!strcmp("gone",    tp)) retc = do_Gone(id);
       else if (!strcmp("pong",    tp)) retc = do_Pong(id);
       else if (!strcmp("load",    tp)) retc = do_Load(id);
       else if (!strcmp("avkb",    tp)) retc = do_AvKb(id);
       else if (!strcmp("suspend", tp)) retc = do_SuRes(id, 0);
       else if (!strcmp("resume",  tp)) retc = do_SuRes(id, 1);
       else if (!strcmp("stage",   tp)) retc = do_StNst(id, 1);
       else if (!strcmp("nostage", tp)) retc = do_StNst(id, 0);
       else if (!strcmp("rst",     tp)) retc = do_RST(id);
       else if (!strcmp("disc",    tp)) retc = do_Disc(id, 1);
       else retc = 1;
       if (retc > 0)
          Say.Emsg("Server", "invalid response from", myNick);
      } while(retc >= 0 && !isOffline);

// Check for permanent errors
//
   if (!isOffline)
      {isOffline = 1;
       if (retc < 0 && Link->LastError())
          Say.Emsg("Server", Link->LastError(), "read response from",
                        Link->Name());
      }

// All done
//
   return retc;
}

/******************************************************************************/
/*                                R e s u m e                                 */
/******************************************************************************/

int XrdOlbServer::Resume(XrdOlbPrepArgs *pargs)   // Static!!!
{

// If we have a prep argument then this is a timed resumption. The caller will
// delete the pargs object if the object has not been rescheduled.
//
    if (pargs) return do_PrepSel(pargs, pargs->Stage);

// Otherwise we are being asked to process all queued prepare arguments. If we
// are an endpoint then we must do the prepare for real. Otherwise, simply
// do a server selection and, if need be, tell the server to stage the file.
//
   do { pargs = XrdOlbPrepArgs::Request();
        if (pargs->endP) {do_PrepAdd4Real(pargs);     delete pargs;}
           else if (!do_PrepSel(pargs, pargs->Stage)) delete pargs;
      } while(1);

// Keep the compiler happy
//
   return 0;
}
  
/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/
  
int XrdOlbServer::Send(const char *buff, int blen)
{
    return (isOffline ? -1 : Link->Send(buff, blen));
}

int  XrdOlbServer::Send(const struct iovec iov[], int iovcnt)
{
    return (isOffline ? -1 : Link->Send(iov, iovcnt));
}

/******************************************************************************/
/*                               s e t N a m e                                */
/******************************************************************************/
  
void XrdOlbServer::setName(XrdNetLink *lnkp, int port)
{
   char buff[512];
   const char *hname = lnkp->Name();
   const char *hnick = lnkp->Nick();

   if (myName)
      if (!strcmp(myName, hname) && port == Port) return;
         else free(myName);

   if (!port) myName = strdup(hname);
      else {sprintf(buff, "%s:%d", hname, port); myName = strdup(buff);}
   Port = port;

   if (myNick) free(myNick);
   if (!port) myNick = strdup(hnick);
      else {sprintf(buff, "%s:%d", hnick, port); myNick = strdup(buff);}
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/* MANAGER LOCAL:                d o _ A v K b                                */
/******************************************************************************/
  
// Server responses to space usage requests are localized to the cell and need
// not be propopagated in any direction.
//
int XrdOlbServer::do_AvKb(char *rid)
{
    char *tp;
    int fdsk, udsk;

// Process: <id> avkb <fsdsk> {<totdsk> | <util>}
//
   if (!(tp = Link->GetToken())
   || XrdOuca2x::a2i(Say, "fs kb value",  tp, &fdsk, 0))
      return 1;
   DiskFree = fdsk;
   if ((tp = Link->GetToken()))
      if (XrdOuca2x::a2i(Say, "util value",  tp, &udsk, 0)) return 1;
         else {if (udsk > 100)
                  {long long Fdsk = static_cast<long long>(fdsk);
                   if ((udsk = 100-(Fdsk*100/udsk)) > 100) udsk=100;
                  }
               DiskTotu = udsk;
              }
   return 0;
}
  
/******************************************************************************/
/*                              d o _ C h m o d                               */
/******************************************************************************/
  
// Manager requests to do a chmod must be forwarded to all subscribers.
//
int XrdOlbServer::do_Chmod(char *rid, int do4real)
{
   EPNAME("do_Chmod")
   char *tp, modearg[16];
   char lclpath[XrdOlbMAX_PATH_LEN+1];
   mode_t mode;
   int rc;

// Process: <id> chmod <mode> <path>
// Respond: n/a
//
   if (!(tp = Link->GetToken())
   || strlcpy(modearg, tp, sizeof(modearg)) >= sizeof(modearg))
      {Say.Emsg("Server", "Mode too long in chmod", tp);
       return 0;
      }

// Get the target name
//
   if (!(tp = Link->GetToken()))
      {Say.Emsg("Server", "chmod path not specified");
       return 0;
      }

// If we have an XMI then pass the request there
//
   if (Xmi_Chmod
   && (!getMode(rid, tp, modearg, mode) || Xmi_Chmod->Chmod(&Req, tp, mode)))
      return 0;

// If are a manager then broadcast the request to every server that might
// might be able to do this operation, then check if we should do it as well.
//
   if (!do4real)
      {if (Manager.ServCnt) Reissue(rid, "chmod ", modearg, tp);
       return 0;
      }

// Convert the mode
//
   if (!(mode = strtol(modearg, 0, 8)))
      {Say.Emsg("Server", "Invalid mode in chmod", modearg, tp);
       return 0;
      }

// Generate the true local path
//
   if (Config.lcl_N2N)
      if (Config.lcl_N2N->lfn2pfn(tp,lclpath,sizeof(lclpath))) return 0;
         else tp = lclpath;

// Attempt to change the mode
//
   if (Config.ProgCH) rc = Config.ProgCH->Run(modearg, tp);
      else if (chmod(tp, mode)) rc = errno;
              else rc = 0;
   if (rc && rc != ENOENT)
       Say.Emsg("Server", errno, "change mode for", tp);
       else DEBUG("rc=" <<rc <<" chmod " <<std::oct <<mode <<std::dec <<' ' <<tp);
   return 0;
}

/******************************************************************************/
/*                              d o _ D e l a y                               */
/******************************************************************************/
  
int XrdOlbServer::do_Delay(char *rid)
{   char respbuff[64];

   return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                               "%s !wait %d\n", rid, Config.SUPDelay));
}

/******************************************************************************/
/* ALL:                          d o _ D i s c                                */
/******************************************************************************/

// When a manager receives a disc response from a server it sends a disc request
// When a server receives a disc request it simply closes the connection.

int XrdOlbServer::do_Disc(char *rid, int sendDisc)
{

// Indicate we have received a disconnect
//
   Say.Emsg("Server", Link->Name(), "requested a disconnect");

// If we must send a disc request, do so now
//
   if (sendDisc) Link->Send("1@0 disc\n", 9);

// Close the link and return an error
//
   isOffline = 1;
   Link->Close(1);
   return -86;
}

/******************************************************************************/
/* MANAGER:                      d o _ G o n e                                */
/******************************************************************************/

// When a manager receives a gone request it is propogated if we are subscribed
// and we have not sent a gone request in the immediate past.
//
int XrdOlbServer::do_Gone(char *rid)
{
   char *tp;
   int   newgone = 0;

// Process: <id> gone <path>
//
   if (!(tp = Link->GetToken())) return 1;

// Update path information (we are not sure we should delete this from the
// prep queue is we are functioning both manager and server, sugh -- we'll see)
//
   if (Instance) newgone = Cache.DelFile(tp, ServMask);
      else Say.Emsg("Server", "gone request ignored from", Name());
   if (Config.DiskSS) PrepQ.Gone(tp);

// If we have no managers and we still have the file or never had it, return
//
   if (!Manager.haveManagers() || !newgone) return 0;

// Back-propogate the gone to all of our managers
//
   Manager.Inform("gone ", 5, tp);

// All done
//
   return 0;
}

/******************************************************************************/
/* MANAGER:                      d o _ H a v e                                */
/******************************************************************************/
  
// When a manager receives a have request it is propogated if we are subscribed
// and we have not sent a have request in the immediate past.
//
int XrdOlbServer::do_Have(char *rid)
{
   char *cmd, *tp;
   int isnew, isrw;

// Process: <id> have {r | w | ?} <path>
//
   if (!(tp = Link->GetToken())) return 1;
   if (*tp == 'r') isrw = 0;
      else if (*tp == 'w') isrw = 1;
              else isrw = -1;
   if (!(tp = Link->GetToken())) return 1;

// Update path information
//
   if (Instance) isnew = Cache.AddFile(tp, ServMask, isrw);
      else {Say.Emsg("Server", "have request ignored from", Name());
            return 0;
           }

// Return if we have no managers or we already informed the managers
//
   if (!Manager.haveManagers() || !isnew) return 0;

// Back-propogate the have to all of our managers
//
   if (!isrw) cmd = (char *)"have r ";
      else if (isrw < 0) cmd = (char *)"have ? ";
              else cmd = (char *)"have w ";
   Manager.Inform(cmd, 7, tp);

// All done
//
   return 0;
}
  
/******************************************************************************/
/* MANAGER LOCAL:                d o _ L o a d                                */
/******************************************************************************/
  
// Server responses to usage requests are local to the cell and never propagated.
//
int XrdOlbServer::do_Load(char *rid)
{
    char *tp;
    int temp, pcpu, pio, pload, pmem, ppag, fdsk, udsk = -1;

// Process: <id> load <cpu> <io> <load> <mem> <pag> <dsk>
//
   if (!(tp = Link->GetToken())) return 1;
   if (XrdOuca2x::a2i(Say, "cpu value",  tp, &pcpu, 0, 100))
      return 1;
   if (!(tp = Link->GetToken())) return 1;
   if (XrdOuca2x::a2i(Say, "io value",   tp, &pio,  0, 100))
      return 1;
   if (!(tp = Link->GetToken())) return 1;
   if (XrdOuca2x::a2i(Say, "load value", tp, &pload,0, 100))
      return 1;
   if (!(tp = Link->GetToken())) return 1;
   if (XrdOuca2x::a2i(Say, "mem value",  tp, &pmem, 0, 100))
      return 1;
   if (!(tp = Link->GetToken())) return 1;
   if (XrdOuca2x::a2i(Say, "pag value",  tp, &ppag, 0, 100))
      return 1;
   if (!(tp = Link->GetToken())) return 1;
   if (XrdOuca2x::a2i(Say, "fs dsk value",  tp, &fdsk, 0))
      return 1;

// The last value is tricky. Old servers returned the total amount of free
// space but new servers return total utilization (0-100). We use the range
// to indicate whether this is a new or old server. Old servers always
// willcalculate based on the sent values, though that is specious since
// the total is truncated at 2TB and most severs have more than that.
//
   if ((tp = Link->GetToken())
   && XrdOuca2x::a2i(Say, "utl dsk value",  tp, &udsk, 0))
      return 1;
   if (udsk > 100)
       {long long Fdsk = static_cast<long long>(fdsk);
        if ((udsk = 100 - (Fdsk*100/udsk)) > 100) udsk = 100;
       }

// Compute actual load value
//
   myLoad = Meter.calcLoad(pcpu, pio, pload, pmem, ppag);
   DiskFree = fdsk;
   DiskTotu = udsk;
   newload = 1;

// If we are also a manager then use this load figure to come up with
// an overall load to report when asked. If we get free space, then we
// must report that now so that we can be selected for allocation.
//
   if (Config.asManager())
      {mlMutex.Lock();
       temp = cpu_load + cpu_load/2;
       cpu_load = (cpu_load + (pcpu > temp ? temp : pcpu ))/2;
       temp = net_load + net_load/2;
       net_load = (net_load + (pio  > temp ? temp : pio  ))/2;
       temp = xeq_load + xeq_load/2;
       xeq_load = (xeq_load + (pload> temp ? temp : pload))/2;
       temp = mem_load + mem_load/2;
       mem_load = (mem_load + (pmem > temp ? temp : pmem ))/2;
       temp = pag_load + pag_load/2;
       pag_load = (pag_load + (ppag > temp ? temp : ppag ))/2;
       if (isRW && DiskFree > dsk_free)
          {temp = dsk_free; dsk_free = DiskFree; dsk_totu = DiskTotu;
           if (!temp)   
              {char respbuff[128];
               Manager.Inform("avkb", 4,
                               respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                               " %d %d\n", dsk_free, dsk_totu));
              }
          }
       mlMutex.UnLock();
      }

// Report new load if need be
//
   if (!Config.LogPerf || logload--) return 0;

// Log it now
//
   {char buff[1024];
    snprintf(buff, sizeof(buff)-1,
            "load=%d; cpu=%d i/o=%d inq=%d mem=%d pag=%d dsk=%d utl=%d",
            myLoad, pcpu, pio, pload, pmem, ppag, fdsk, udsk);
    Say.Emsg("Server", Name(), buff);
    if ((logload = Config.LogPerf)) logload--;
   }

   return 0;
}
  
/******************************************************************************/
/*                             d o _ L o c a t e                              */
/******************************************************************************/

int XrdOlbServer::do_Locate(char *rid, const char *path,
                            SMask_t hfVec, SMask_t rwVec)
{
   EPNAME("do_Locate";)
   XrdOlbSInfo *sP, *pP;
   static const int Skip = (OLB_SERVER_DISABLE | OLB_SERVER_OFFLINE);
   char outbuff[32*64], *oP;

// List the servers
//
   if (!hfVec || !(sP = Manager.ListServers(hfVec, OLB_LS_IPO)))
      {Link->Send(outbuff, snprintf(outbuff, sizeof(outbuff)-1,
             "%s !err ENOENT No servers have the file", rid));
       DEBUG("Path find failed for locate " <<path);
       return 0;
      }

// Insert prefix into the buffer
//
   oP = outbuff + sprintf(outbuff, "%s !data ", rid);

// format out the request as follows:                   
// 01234567810123456789212345678
// xy[::123.123.123.123]:123456
//
   while(sP)
        {if (sP->Status & Skip) continue;
         *oP++ = (sP->Status & OLB_SERVER_ISMANGR ? 'M' : 'S');
         *oP++ = (sP->Mask & rwVec                ? 'w' : 'r');
         strcpy(oP, "[::"); oP += 3;
         oP += XrdNetDNS::IP2String(sP->IPAddr, 0, oP, 24); // We're cheating
         *oP++ = ']'; *oP++ = ':';
         oP += sprintf(oP, "%d", sP->Port);
         pP = sP; 
         if ((sP = sP->next)) *oP++ = ' ';
         delete pP;
        }

// Send of the result
//
   *oP = '\0';
   Link->Send(outbuff, oP-outbuff);
   return 0;
}
  
/******************************************************************************/
/*                              d o _ M k d i r                               */
/******************************************************************************/
  
// Manager requests to do a mkdir must be forwarded to all subscribers.
//
int XrdOlbServer::do_Mkdir(char *rid, int do4real)
{
   EPNAME("do_Mkdir";)
   char *tp, modearg[16];
   char lclpath[XrdOlbMAX_PATH_LEN+1];
   mode_t mode;
   int rc;

// Process: <id> mkdir <mode> <path>
// Respond: n/a
//
   if (!(tp = Link->GetToken())
   || strlcpy(modearg, tp, sizeof(modearg)) >= sizeof(modearg))
      {Say.Emsg("Server", "Mode too long in mkdir", tp);
       return 0;
      }

// Get the directory name
//
   if (!(tp = Link->GetToken()))
      {Say.Emsg("Server", "mkdir directory not specified");
       return 0;
      }

// If we have an XMI then pass the request there
//
   if (Xmi_Mkdir 
   && (!getMode(rid, tp, modearg, mode) || Xmi_Mkdir->Mkdir(&Req, tp, mode)))
      return 0;

// If we have subsscribers then broadcast the request to every server that
// might be able to do this operation
//
   if (!do4real)
      {if (Manager.ServCnt) Reissue(rid, "mkdir ", modearg, tp);
       return 0;
      }

// Convert the mode
//
   if (!(mode = strtol(modearg, 0, 8)))
      {Say.Emsg("Server", "Invalid mode in mkdir", modearg, tp);
       return 0;
      }

// Generate the true local path
//
   if (Config.lcl_N2N)
      if (Config.lcl_N2N->lfn2pfn(tp,lclpath,sizeof(lclpath))) return 0;
         else tp = lclpath;

// Attempt to create the directory
//
   if (Config.ProgMD) rc = Config.ProgMD->Run(modearg, tp);
      else if (mkdir(tp, mode)) rc = errno;
              else rc = 0;
   if (rc) Say.Emsg("Server", rc, "create directory", tp);
      else DEBUG("rc=" <<rc <<" mkdir " <<std::oct <<mode <<std::dec <<' ' <<tp);
   return 0;
}
  
/******************************************************************************/
/*                             d o _ M k p a t h                              */
/******************************************************************************/
  
  
// Manager requests to do a mkpath must be forwarded to all subscribers.
//
int XrdOlbServer::do_Mkpath(char *rid, int do4real)
{
   EPNAME("do_Mkpath";)
   char *tp, modearg[16];
   char lclpath[XrdOlbMAX_PATH_LEN+1];
   mode_t mode;
   int rc;

// Process: <id> mkpath <mode> <path>
// Respond: n/a
//
   if (!(tp = Link->GetToken())
   || strlcpy(modearg, tp, sizeof(modearg)) >= sizeof(modearg))
      {Say.Emsg("Server", "Mode too long in mkpath", tp);
       return 0;
      }

// Get the directory name
//
   if (!(tp = Link->GetToken()))
      {Say.Emsg("Server", "mkpath directory not specified");
       return 0;
      }

// If we have an XMI then pass the request there
//
   if (Xmi_Mkpath
   && (!getMode(rid, tp, modearg, mode) || Xmi_Mkpath->Mkpath(&Req, tp, mode)))
      return 0;

// If we have subsscribers then broadcast the request to every server that
// might be able to do this operation
//
   if (!do4real)
      {if (Manager.ServCnt) Reissue(rid, "mkpath ", modearg, tp);
       return 0;
      }

// Convert the mode
//
   if (!(mode = strtol(modearg, 0, 8)))
      {Say.Emsg("Server", "Invalid mode in mkpath", modearg, tp);
       return 0;
      }

// Generate the true local path
//
   if (Config.lcl_N2N)
      if (Config.lcl_N2N->lfn2pfn(tp,lclpath,sizeof(lclpath))) return 0;
         else tp = lclpath;

// Attempt to create the directory path
//
   if (Config.ProgMD) rc = Config.ProgMP->Run(modearg, tp);
      else            rc = Mkpath(tp, mode);
   if (rc) Say.Emsg("Server", rc, "create path", tp);
      else DEBUG("rc=" <<rc <<" mkpath " <<std::oct <<mode <<std::dec <<' ' <<tp);
   return 0;
}

/******************************************************************************/
/*                                 d o _ M v                                  */
/******************************************************************************/
  
// Manager requests to do an mv must be forwarded to all subscribers.
//
int XrdOlbServer::do_Mv(char *rid, int do4real)
{
   EPNAME("do_Mv")
   char *tp;
   char old_lfnpath[XrdOlbMAX_PATH_LEN+1];
   char old_lclpath[XrdOlbMAX_PATH_LEN+1];
   char new_lclpath[XrdOlbMAX_PATH_LEN+1];
   int rc;

// Process: <id> mv <old_name> <new_name>
// Respond: n/a

// Get the old name
//
   if (!(tp = Link->GetToken()))
      {Say.Emsg("Server", "mv old path not specified");
       return 0;
      }

// If we have an XMI or are virtual then we must save the original path name
//
   if (!do4real || Xmi_Rename)
      {if (strlcpy(old_lfnpath,tp,sizeof(old_lfnpath)) > XrdOlbMAX_PATH_LEN)
          {Say.Emsg("Server", "mv old path too long", tp);
           return 0;
          }
      }

// Generate proper old path name
//
   if (do4real && Config.lcl_N2N)
      {if (Config.lcl_N2N->lfn2pfn(tp,old_lclpath,sizeof(old_lclpath))) 
          return 0;
      }
      else if (strlcpy(old_lclpath,tp,sizeof(old_lclpath)) > XrdOlbMAX_PATH_LEN)
              {Say.Emsg("Server", "mv old path too long", tp);
               return 0;
              }

// Get the new name
//
   if (!(tp = Link->GetToken()))
      {Say.Emsg("Server", "mv new path not specified");
       return 0;
      }

// If we have an XMI then pass the request there. This only works when
// we are a manager. We still have to figure out what to do for servers.
//
   if (Xmi_Rename)
      {strcpy(Info.ID, rid);  // Gauranteed to fit
       if (Xmi_Rename->Rename(&Req, old_lfnpath, tp)) return 0;
      }

// If we have subscribers then broadcast the request to every server that
// might be able to do this operation
//
   if (!do4real)
      {if (Manager.ServCnt) Reissue(rid, "mv ", 0, old_lfnpath, tp);
       return 0;
      }

// Generate the true local path for new name
//
   if (Config.lcl_N2N)
      if (Config.lcl_N2N->lfn2pfn(tp,new_lclpath,sizeof(new_lclpath))) 
         return 0;
         else tp = new_lclpath;

// Attempt to rename the file
//
   if (Config.ProgMV) rc = Config.ProgMV->Run(old_lclpath, tp);
      else if (rename(old_lclpath, tp)) rc = errno;
              else rc = 0;
   if (rc) Say.Emsg("Server", rc, "rename", old_lclpath);
      else DEBUG("rc=" <<rc <<" mv " <<old_lclpath <<' ' <<tp);

   return 0;
}
  
/******************************************************************************/
/* SERVER LOCAL:                 d o _ P i n g                                */
/******************************************************************************/
  
// Ping requests from a manager are local to the cell and never propagated.
//
int XrdOlbServer::do_Ping(char *rid)
{
    char respbuff[64];

// Process: <id> ping
// Respond: <id> pong
//
   return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                               "%s pong\n", rid));
}
  
/******************************************************************************/
/*                            d o _ P r e p A d d                             */
/******************************************************************************/
  
int XrdOlbServer::do_PrepAdd(char *rid, int server)
{
   XrdOlbPrepArgs *pargs = new XrdOlbPrepArgs(server);
   char *Line;

// Process: <id> prepadd <reqid> <usr> <prty> <mode> <path>\n
// Respond: No response.
//

// Return the previous token so that we can get the whole line back
//
   Link->RetToken();
   Link->GetToken(&Line);
   if (!Line)
      {Say.Emsg("Server", "invalid prepare request from", Name());
       delete pargs; return 1;
      }
   Line = pargs->data = strdup(Line);

// Get the request id and whether we need to stage the file
//
   if (!(pargs->reqid = prepScan(&Line, pargs, "no prep request id"))) return 1;
   pargs->Stage = (*(pargs->reqid) != '*');

// Get userid
//
   if (!(pargs->user = prepScan(&Line, pargs, "no prep user"))) return 1;

// Get priority
//
   if (!(pargs->prty = prepScan(&Line, pargs, "no prep prty"))) return 1;

// Get mode
//
   if (!(pargs->mode = prepScan(&Line, pargs, "no prep mode"))) return 1;

// Get path
//
   if (!(pargs->path = prepScan(&Line, pargs, "no prep path"))) return 1;

// Queue this request for async processing
//
   pargs->Queue(); 
   return 0;
}

/******************************************************************************/
/*                       d o _ P r e p A d d 4 R e a l                        */
/******************************************************************************/
  
int XrdOlbServer::do_PrepAdd4Real(XrdOlbPrepArgs *pargs)  // Static!!!
{
   EPNAME("do_PrepAdd4Real");

// Check if this file is not online, prepare it
//
   if (!isOnline(pargs->path))
      {DEBUG("Preparing " <<pargs->reqid <<' ' <<pargs->user <<' ' <<pargs->prty
                          <<' ' <<pargs->mode <<' ' <<pargs->path);
       if (!Config.DiskSS)
          Say.Emsg("Server", "staging disallowed; ignoring prep",
                          pargs->user, pargs->reqid);
          else if (!Xmi_Prep 
               ||  !Xmi_Prep->Prep(pargs->reqid, pargs->path,
                                  (index(pargs->mode, 'w') ? XMI_RW:0)))
                  PrepQ.Add(*pargs);
       return 0;
      }

// File is already online, so we are done
//
   Inform("avail", pargs);
   return 0;
}
  
/******************************************************************************/
/*                            d o _ P r e p D e l                             */
/******************************************************************************/
  
int XrdOlbServer::do_PrepDel(char *rid, int server)
{
   EPNAME("do_PrepDel")
   char *tp;
   BUFF(2048);

// Process: <id> prepcan <reqid>
// Respond: No response.
//

// Get the request id
//
   if (!(tp = Link->GetToken()))
      {Say.Emsg("Server", "no prepcan request id from", Name());
       return 1;
      }

// Do a callout to the external manager if we have one
//
   if (Xmi_Prep &&  Xmi_Prep->Prep(tp, "", XMI_CANCEL)) return 0;

// If this is a server call, do it for real
//
   if (server)
      {if (!Config.DiskSS) {DEBUG("Ignoring cancel prepare " <<tp);}
          else {DEBUG("Canceling prepare " <<tp);
                PrepQ.Del(tp);
               }
       return 0;
      }

// Cancel the request. Since we don't know where this went, inform all
// subscribers about this cancellation.
//
  if (Manager.ServCnt) 
     {DEBUG("Forwarding prepare cancel " <<tp);
      Reissue(rid, "prepdel ", 0, tp);
     }
  return 0;
}
  
/******************************************************************************/
/*                            d o _ P r e p S e l                             */
/******************************************************************************/
  
int XrdOlbServer::do_PrepSel(XrdOlbPrepArgs *pargs, int stage) // Static!!!
{
   EPNAME("do_PrepSel")
   BUFF(2048);
   XrdOlbPInfo pinfo;
   XrdOlbCInfo cinfo;
   char ptc, hbuff[512];
   int retc, needrw, Osel;
   SMask_t amask, smask, pmask;

// Determine mode
//
   if (index(pargs->mode, (int)'w'))
           {needrw = 1; ptc = 'w'; Osel = OLB_needrw;}
      else {needrw = 0; ptc = 'r'; Osel = 0;}

// Do a callout to the external manager if we have one
//
   if (Xmi_Prep)
      {int opts = (Osel & OLB_needrw ? XMI_RW : 0);
       if (Xmi_Prep->Prep(pargs->reqid, pargs->path, opts)) return 0;
      }

// Find out who serves this path
//
   if (!Cache.Paths.Find(pargs->path, pinfo)
   || (amask = (needrw ? pinfo.rwvec : pinfo.rovec)) == 0)
      {DEBUG("Path find failed for " <<pargs->reqid <<' ' <<ptc <<' ' <<pargs->path);
       return Inform("unavail", pargs);
      }

// First check if we have seen this file before. If not, broadcast a lookup
// to all relevant servers. Note that even if the caller wants the file in
// r/o mode we will ask both r/o and r/w servers for the file.
//
   if (!(retc = Cache.GetFile(pargs->path, cinfo))
   ||  cinfo.deadline || (cinfo.sbvec != 0))
      {if (!retc)
          {DEBUG("Searching for " <<pargs->path);
           Cache.AddFile(pargs->path, 0, 0, Config.LUPDelay);
           Manager.Broadcast(pinfo.rovec, buff, snprintf(buff, sizeof(buff)-1,
                           "%s state %s\n", Config.MsgGID, pargs->path));
          } else {
           if (cinfo.sbvec != 0)         // Bouncing server
              {Cache.DelFile(pargs->path,cinfo.sbvec,Config.LUPDelay);
               Manager.Broadcast(cinfo.sbvec, buff, snprintf(buff,sizeof(buff)-1,
                                "%s state %s\n",Config.MsgGID,pargs->path));
              }
          }
       if (!stage) return 0;
       DEBUG("Rescheduling lookup in " <<Config.LUPDelay <<" seconds");
       Sched->Schedule((XrdJob *)pargs, Config.LUPDelay+time(0));
       return 1;
      }

// Compute the primary and secondary selections:
// Primary:   Servers who already have the file
// Secondary: Servers who don't have the file but can stage it in
//
   pmask = (needrw ? cinfo.hfvec & pinfo.rwvec: cinfo.hfvec);
   smask = amask & pinfo.ssvec;   // Alternate selection

// Select a server (do not select any peers)
//
   if (!(pmask | smask)) retc = -1;
      else if (!(retc = Manager.SelServer(Osel, pargs->path, pmask, smask, 
                        hbuff, pargs->Msg, pargs->prepMsg())))
              {DEBUG(hbuff <<" prepared " <<pargs->reqid <<' ' <<pargs->path);
               return 0;
              }

// We failed check if we should reschedule this
//
   if (retc > 0)
      {Sched->Schedule((XrdJob *)pargs, retc+time(0));
       DEBUG("Prepare delayed " <<retc <<" seconds");
       return 1;
      }

// We failed, terminate the request
//
   DEBUG("No servers available to prepare "  <<pargs->reqid <<' ' <<pargs->path);
   Inform("unavail", pargs);
   return 0;
}
  
/******************************************************************************/
/* MANAGER LOCAL:                d o _ P o n g                                */
/******************************************************************************/
  
// Server responses to a ping are local to the cell and never propagated.
//
int XrdOlbServer::do_Pong(char *rid)
{
// Process: <id> pong
// Reponds: n/a

// Simply indicate a ping has arrived.
//
   pingpong = 1;
   return 0;
}
  
/******************************************************************************/
/*                                 d o _ R m                                  */
/******************************************************************************/
  
// Manager requests to do an rm must be forwarded to all subscribers.
//
int XrdOlbServer::do_Rm(char *rid, int do4real)
{
   EPNAME("do_Rm")
   char *tp;
   char lclpath[XrdOlbMAX_PATH_LEN+1];
   int rc;

// Process: <id> rm <path>
// Respond: n/a

// Get the path
//
   if (!(tp = Link->GetToken()))
      {Say.Emsg("Server", "rm path not specified");
       return 0;
      }

// If we have an XMI then pass the request there.
//
   if (Xmi_Remove)
      {strcpy(Info.ID, rid);  // Gauranteed to fit
       if (Xmi_Remove->Remove(&Req, tp)) return 0;
      }

// If we have subscribers then broadcast the request to every server that
// might be able to do this operation
//
   if (!do4real)
      {if (Manager.ServCnt) Reissue(rid, "rm ", 0, tp);
       return 0;
      }

// Generate the true local path for name
//
   if (Config.lcl_N2N)
      if (Config.lcl_N2N->lfn2pfn(tp,lclpath,sizeof(lclpath))) return 0;
         else tp = lclpath;

// Attempt to remove the file
//
   if (Config.ProgRM) rc = Config.ProgRM->Run(tp);
      else if (unlink(tp)) rc = errno;
              else rc = 0;
   if (rc && rc != ENOENT) Say.Emsg("Server", rc, "remove", tp);
      else {DEBUG("rc=" <<rc <<" rm " <<tp);}

   return 0;
}
  
/******************************************************************************/
/*                              d o _ R m d i r                               */
/******************************************************************************/
  
// Manager requests to do an rmdir must be forwarded to all subscribers.
//
int XrdOlbServer::do_Rmdir(char *rid, int do4real)
{
   EPNAME("do_Rmdir")
   char lclpath[XrdOlbMAX_PATH_LEN+1];
   char *tp;
   int rc;

// Process: <id> rmdir <path>
// Respond: n/a

// Get the path
//
   if (!(tp = Link->GetToken()))
      {Say.Emsg("Server", "rmdir path not specified");
       return 0;
      }

// If we have an XMI then pass the request there.
//
   if (Xmi_Remdir)
      {strcpy(Info.ID, rid);  // Gauranteed to fit
       if (Xmi_Remdir->Remdir(&Req, tp)) return 0;
      }

// If we have subscribers then broadcast the request to every server that
// might be able to do this operation
//
   if (!do4real)
      {if (Manager.ServCnt) Reissue(rid, "rmdir ", 0, tp);
       return 0;
      }

// Generate the true local path for name
//
   if (Config.lcl_N2N)
      if (Config.lcl_N2N->lfn2pfn(tp,lclpath,sizeof(lclpath))) return 0;
         else tp = lclpath;

// Attempt to remove the directory
//
   if (Config.ProgRD) rc = Config.ProgRD->Run(tp);
      else if (rmdir(tp)) rc = errno;
              else rc = 0;
   if (rc && errno != ENOENT) Say.Emsg("Server", errno, "remove", tp);
      else DEBUG("rc=" <<errno <<" rmdir " <<tp);

   return 0;
}

/******************************************************************************/
/* MANAGER:                       d o _ R S T                                 */
/******************************************************************************/

// An rst response is received when a subscribed supervisor adds a new server.
// This causes all cache lines for the supervisor to be marked suspect. Also,
// the RST request is propagated to all of our managers.
//
int XrdOlbServer::do_RST(char *rid)
{

// First propagate the RST to our managers
//
   Manager.Reset();

// Now invalidate our cache lines
//
   Cache.Bounce(ServMask);
   return 0;
}
  
/******************************************************************************/
/* MANAGER LOCAL:              d o _ S e l e c t                              */
/******************************************************************************/
  
// A select request comes from a redirector and is handled locally within the
// cell. This may cause "state" requests to be broadcast to subscribers.
//
int XrdOlbServer::do_Select(char *rid, int refresh)
{
   EPNAME("do_Select")
   BUFF(2048);
   XrdOlbRRQInfo *InfoP = &Info;
   XrdOlbPInfo pinfo;
   XrdOlbCInfo cinfo;
   const char *amode;
   char *tp, ptc, hbuff[512];
   int n, dowt = 0, Osel, retc, needrw;
   int qattr = 0, resonly = 0, newfile = 0, dotrunc = 0, dolocate = 0;
   SMask_t amask, smask, pmask, nmask = ~SMask_t(0);

// Process: <id> select[s] {c | d | r | w | s | t | x | z] [-host] <path>

// Note: selects - requests a cache refresh for <path>
//             c - file will be created
//             d - file will be created or truncated
//             r - file will only be read
//             w - file will be read and writen
//             s - only stat information will be obtained
//             x - only stat information will be obtained (file must be resident)
//             y - only loc  information will be obtained (do asap)
//             z - only loc  information will be obtained
//             - - the host failed to deliver the file.

// Reponds: ?err  <msg>
//          !try  <host>
//          !wait <sec>

// Pick up Parameters
//
   if (!(tp = Link->GetToken()) || strlen(tp) != 1) return 1;
   ptc = *tp;
   switch(*tp)
        {case 'r': Osel = 0;           amode = "read";  break;
         case 's': Osel = 0;           amode = "read";  qattr   = 1; break;
         case 'w': Osel = OLB_needrw;  amode = "write"; break;
         case 'x': Osel = 0;           amode = "read";  resonly = 1;
                                                        qattr   = 1; break;
         case 'y': Osel = 0;           amode = "read";  dolocate= 1; break;
         case 'z': Osel = 0; InfoP=0;  amode = "read";  dolocate= 1; break;
         case 'c': Osel = OLB_newfile; amode = "write"; newfile = 1; break;
         case 'd': Osel = OLB_newfile; amode = "write"; newfile = 1;
                                                        dotrunc = 1; break;
         case 't': Osel = OLB_newfile; amode = "write"; dotrunc = 1; break;
         default:  return 1;
        }
   if (!(tp = Link->GetToken())) return 1;

// Check if an avoid host is here
//
   if (*tp == '-')
      {unsigned int IPaddr;
       if (XrdNetDNS::Host2IP(tp+1, &IPaddr)) 
          {nmask = ~Manager.getMask(IPaddr); InfoP = 0;}
       if (!(tp = Link->GetToken())) return 1;
       InfoP = 0;
      }

// Do a callout to the external manager if we have one
//
      if (qattr) {if (Xmi_Stat && Xmi_Stat->Stat(&Req, tp)) return 0;}
         else if (Xmi_Select) 
                 {int opts = (Osel & OLB_needrw ? XMI_RW : 0);
                  if (newfile) opts |= XMI_NEW;
                  if (dotrunc) opts |= XMI_TRUNC;
                  if (dolocate)opts |= XMI_LOCATE;
                  if (Xmi_Select->Select(&Req, tp, opts)) return 0;
                 }

// Find out who serves this path
//
   needrw = Osel & (OLB_needrw | OLB_newfile); Osel |= OLB_peersok;
   if (!Cache.Paths.Find(tp, pinfo)
   || (amask = ((needrw ? pinfo.rwvec : pinfo.rovec) & nmask)) == 0)
      {Link->Send(buff, snprintf(buff, sizeof(buff)-1,
             "%s !err ENOENT No servers have %s access to the file",rid,amode));
       DEBUG("Path find failed for select " <<ptc <<' ' <<tp);
       return 0;
      }

// Insert the request ID into the RRQ info structure in case we need to wait
//
   strcpy(Info.ID, rid);  // Gauranteed to fit
   Info.isLU = dolocate;
   Info.Arg  = pinfo.rwvec;

// First check if we have seen this file before. If so, get primary selections.
//
   if (refresh) {retc = 0; pmask = 0;}
      else if (!(retc = Cache.GetFile(tp,cinfo,needrw,InfoP))) pmask = 0;
              else pmask = (needrw ? cinfo.hfvec & pinfo.rwvec
                                   : cinfo.hfvec & nmask);

// We didn't find the file or a refresh is wanted (easy case). Client must wait.
//
   if (!retc)
      {Cache.AddFile(tp, 0, needrw, Config.LUPDelay, InfoP);
       Manager.Broadcast(pinfo.rovec, buff, snprintf(buff, sizeof(buff)-1,
                          "%s stat%c %s\n", Config.MsgGID,
                          (refresh ? 'f' : 'e'), tp));
       if (InfoP && Info.Key) return 0; // Placed in pending state
       dowt = 1;
      } else

// File was found but either a query is in progress (client must wait)
// or we have a server bounce (client waits if no alternative is available).
// Unfortunately, fast redirects are bypassed when servers bounce.
//
      {if (cinfo.sbvec != 0)         // Bouncing server
          {dowt = (pmask == 0);
           Cache.DelFile(tp, cinfo.sbvec, Config.LUPDelay);
           Manager.Broadcast(cinfo.sbvec,buff,snprintf(buff,sizeof(buff)-1,
                              "%s state %s\n", Config.MsgGID, tp));
          }
       if (cinfo.deadline) 
          if (InfoP && Info.Key) return 0; // Placed in pending queue
             else dowt = 1;                // Query  in progress, full wait
      }

// If the client has to wait now, delay the client and return
//
   if (dowt)
      {Link->Send(buff,sprintf(buff,"%s !wait %d\n",rid,Config.LUPDelay));
       DEBUG("Lookup delay " <<Name() <<' ' <<Config.LUPDelay);
       return 0;
      }

// If this is a locate request, perform it now
//
   if (dolocate) return do_Locate(rid, tp, cinfo.hfvec, pinfo.rwvec);

// Compute the primary and secondary selections:
// Primary:   Servers who already have the file (computed above)
// Secondary: Servers who don't have the file but can get it
//
   if (resonly) smask = 0;
      else smask = (newfile ? nmask & pinfo.rwvec
                            : amask & pinfo.ssvec); // Alt selection

// Select a server
//
   if (!(pmask | smask)) retc = -1;
      else if (!(retc = Manager.SelServer(Osel, tp, pmask, smask, hbuff)))
              {DEBUG("Redirect " <<Name() <<" -> " <<hbuff <<" for " <<tp);
               redr_iov[0].iov_base = rid;   redr_iov[0].iov_len = strlen(rid);
               n = strlen(hbuff); hbuff[n] = '\n';
               redr_iov[2].iov_base = hbuff; redr_iov[2].iov_len = n+1;
               Link->Send(redr_iov, redr_iov_cnt);
               return 0;
              }

// We failed and must delay or terminate the request. If the request needs to
// be delayed because there are not enough servers then we must delete the
// cache line to force a cache refresh when we get enough servers.
//
   if (retc > 0)
      {Link->Send(buff, sprintf(buff, "%s !wait %d\n", rid, retc));
       DEBUG("Select delay " <<Name() <<' ' <<retc);
      } else {
       Link->Send(buff, snprintf(buff, sizeof(buff)-1,
                "%s !err ENOENT No servers are available to %s the file.\n",rid,amode));
       DEBUG("No servers available to " <<ptc <<' ' <<tp);
      }

// All done
//
   return 0;
}
  
/******************************************************************************/
/* SERVER:                      d o _ S p a c e                               */
/******************************************************************************/
  
// Manager space requests are local to the cell and never propagated.
//
int XrdOlbServer::do_Space(char *rid)
{
   char respbuff[128];
   int maxfr, tutil;

// Process: <id> space
// Respond: <id> avkb  <numkb>
//
   if (Config.asManager())
      return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                        "%s avkb %d %d\n", rid, dsk_free, dsk_totu));

   maxfr = Meter.FreeSpace(tutil);
   return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                "%s avkb %d %d\n", rid, maxfr, tutil));
}
  
/******************************************************************************/
/* SERVER:                      d o _ S t a t e                               */
/******************************************************************************/
  
// State requests from a manager are rebroadcast to all relevant subscribers.
//
int XrdOlbServer::do_State(char *rid,int reset)
{
   char *tp, respbuff[2048];

// Process: <id> state <path>
//          <id> statf <path>
// Respond: <id> {gone | have {r | w | s | ?}} <path>
//
   if (!(tp = Link->GetToken())) return 1;
   isKnown = 1;

// If we are a manager then check for the file in the local cache
//
   if (isMan && do_StateFWD(tp, reset))
      return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
             "%s have %s %s\n", rid, Config.PathList.Type(tp), tp));
   if (!Manager.hasData) return 0;

// Do a stat, respond if we have the file
//
   if (isOnline(tp, 0, Link))
      return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
             "%s have %s %s\n", rid, Config.PathList.Type(tp), tp));
   return 0;
}
  
/******************************************************************************/
/* SUPER:                    d o _ S t a t e F W D                            */
/******************************************************************************/
  
int XrdOlbServer::do_StateFWD(char *tp, int reset)
{
   EPNAME("do_StateFWD");
   BUFF(2048);
   XrdOlbPInfo pinfo;
   XrdOlbCInfo cinfo;
   int docr, dowt, retc;
   SMask_t pmask = 0;

// Find out who serves this path
//
   if (!Cache.Paths.Find(tp, pinfo) || pinfo.rovec == 0)
      {DEBUG("Path find failed for state " <<tp);
       return 0;
      }

// First check if we have seen this file before. If so, get primary selections.
//
   if (reset) retc = 0;
      else if ((retc = Cache.GetFile(tp, cinfo))) pmask = cinfo.hfvec;

// If we didn't find the file, or it's being searched for, then return failure.
// Otherwise, we will ask the relevant servers if they have the file. We return
//
   dowt = (!retc || cinfo.deadline);
   docr = (retc && (cinfo.sbvec != 0));
   if (dowt || docr)
      {if (!retc) Cache.AddFile(tp, 0, 0, Config.LUPDelay);
          else Cache.DelFile(tp, cinfo.sbvec, (dowt ? Config.LUPDelay : 0));
       Manager.Broadcast((retc ? cinfo.sbvec : pinfo.rovec), buff,
                          snprintf(buff, sizeof(buff)-1,
                          "%s stat%c %s\n", Config.MsgGID, 
                          (reset ? 'f' : 'e'), tp));
      }

// Return true if anyone has the file at this point
//
   return (pmask != 0);
}

/******************************************************************************/
/* ANY:                         d o _ S t a t s                               */
/******************************************************************************/
  
// We punt on stats requests as we have no way to export them anyway.
//
int XrdOlbServer::do_Stats(char *rid, int full)
{
   static XrdSysMutex StatsData;
   static int    statsz = 0;
   static int    statln = 0;
   static char  *statbuff = 0;
   static time_t statlast = 0;
   int rc;
   time_t tNow;

// Allocate buffer if we do not have one
//
   StatsData.Lock();
   if (!statsz || !statbuff)
      {statsz    = Manager.Stats(0,0);
       statbuff = (char *)malloc(statsz);
      }

// Check if only the size is wanted
//
   if (!full)
      {char respbuff[32];
       StatsData.UnLock();
       return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                                   "%d\n", statsz));
      }

// Get full statistics if enough time has passed
//
   tNow = time(0);
   if (statlast+9 >= tNow)
      {statln = Manager.Stats(statbuff, statsz); statlast = tNow;}

// Send response
//
   if (statln) rc = Link->Send(statbuff, statln);
      else     rc = Link->Send("\n", 1);

// All done
//
   StatsData.UnLock();
   return rc;
}
  
/******************************************************************************/
/* MANAGER:                     d o _ S t N s t                               */
/******************************************************************************/
  
// When a manager receives a stage/nostage request, the result is propagated
// to upper-level managers only if the summary state has changed.
//
int XrdOlbServer::do_StNst(char *rid, int Stage)
{
    const char *why;

    if (Stage)
       {if (!isNoStage) return 0;
        isNoStage = 0;
        if (!isOffline && !isDisable) why = "staging resumed";
           else why = isOffline ? "offlined" : "disabled";
       } else if (isNoStage) return 0;
                 else {why = "staging suspended"; isNoStage = 1;}

    if (isMan) OlbState.Calc(Stage ? 1 : -1, 0, 0);

    Say.Emsg("Server", Name(), why);
    return 0;
}
  
/******************************************************************************/
/* MANAGER:                     d o _ S u R e s                               */
/******************************************************************************/
  
// When a manager receives a suspend/resume request, the result is propagated
// to upper-level managers only if the summary state has changed.
//
int XrdOlbServer::do_SuRes(char *rid, int Resume)
{
    const char *why;

    if (Resume) 
       {if (!isSuspend) return 0;
        isSuspend = 0;
        if (!isOffline && !isDisable) why = "service resumed";
           else why = isOffline ? "offlined" : "disabled";
       } else if (isSuspend) return 0;
                 else {why = "service suspended"; isSuspend = 1;}

    if (isMan) OlbState.Calc(Resume ? -1 : 1, 1, 1);

    Say.Emsg("Server", Name(), why);
    return 0;
}

/******************************************************************************/
/* SERVER LOCAL:                  d o _ T r y                                 */
/******************************************************************************/

// Try requests from a manager indicate that we are being displaced and should
// hunt for another manager. The request provides hints as to where to try.
//
int XrdOlbServer::do_Try(char *rid)
{
   char *tp;
   unsigned int ipaddr = Link->Addr();

// Delete any additions from this manager
//
   myMans.Del(ipaddr);

// Add all the alternates to our alternate list
//
   while((tp = Link->GetToken()))
         myMans.Add(ipaddr, tp, Config.PortTCP, myLevel);

// Close the link and reurn an error
//
   isOffline = 1;
   Link->Close(1);
   return -1;
}
  
/******************************************************************************/
/* SERVER LOCAL:                d o _ U s a g e                               */
/******************************************************************************/
  
// Usage requests from a manager are local to the cell and never propagated.
//
int XrdOlbServer::do_Usage(char *rid)
{
    char respbuff[512];
    int  maxfr, tutil;

// Process: <id> usage
// Respond: <id> load <cpu> <io> <load> <mem> <pag> <dskfree> <dskutil>
//
   if (Config.asManager())
      return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                        "%s load %d %d %d %d %d %d %d\n", rid,
                        cpu_load, net_load, xeq_load, mem_load, pag_load,
                        dsk_free, dsk_totu));

   if (Meter.isOn())
   return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                     "%s load %s\n", rid, Meter.Report()));

   maxfr = Meter.FreeSpace(tutil);
   return Link->Send(respbuff, snprintf(respbuff, sizeof(respbuff)-1,
                     "%s load 0 0 0 0 0 %d %d\n", rid, maxfr, tutil));
}

/******************************************************************************/
/*                               g e t M o d e                                */
/******************************************************************************/

int XrdOlbServer::getMode(const char *rid, const char *path,
                          const char *modearg, mode_t &mode)
{

// Copy the request id
//
   strcpy(Info.ID, rid);  // Gauranteed to fit

// Convert the mode argument
//
   if (!(mode = strtol(modearg, 0, 8)))
      {Say.Emsg("Server", "Invalid mode in chmod", modearg, path);
       Req.Reply_Error("EINVAL", "Invalid mode");
       return 0;
      }

// All is well
//
   return 1;
}

/******************************************************************************/
/*                                I n f o r m                                 */
/******************************************************************************/
  
int XrdOlbServer::Inform(const char *cmd, XrdOlbPrepArgs *pargs)
{
   EPNAME("Inform")
   char *mdest, *minfo;

// See if user wants a response
//
   if (!index(pargs->mode, (int)'n')
   ||  strcmp("udp://", pargs->mode)
   ||  !Relay)
      {DEBUG(cmd <<' ' <<pargs->reqid <<" not sent to " <<pargs->user);
       return 0;
      }

// Extract out destination and argument
//
   mdest = pargs->mode+6;
   if ((minfo = index(pargs->mode, (int)'/')))
      {*minfo = '\0'; minfo++;}
   if (!minfo || !*minfo) minfo = (char *)"*";
   DEBUG("Sending " <<mdest <<": " <<cmd <<' '<<pargs->reqid <<' ' <<minfo);

// Send the message and return
//
   Relay->Send(mdest, pargs->Msg, pargs->prepMsg(cmd, minfo));
   return 0;
}

/******************************************************************************/
/*                              i s O n l i n e                               */
/******************************************************************************/
  
int XrdOlbServer::isOnline(char *path, int upt, XrdNetLink *myLink) // Static!!!
{
   struct stat buf;
   struct utimbuf times;
   char *lclpath, lclbuff[XrdOlbMAX_PATH_LEN+1];

// Generate the true local path
//
   lclpath = path;
   if (Config.lcl_N2N)
      if (Config.lcl_N2N->lfn2pfn(lclpath,lclbuff,sizeof(lclbuff))) return 0;
         else lclpath = lclbuff;

// Do a stat
//
   if (stat(lclpath, &buf))
      if (Config.DiskSS && PrepQ.Exists(path)) return 1;
         else return 0;

// Update access time if so requested but only if this is a file
//
   if ((buf.st_mode & S_IFMT) == S_IFREG)
      {if (upt)
          {times.actime = time(0);
           times.modtime = buf.st_mtime;
           utime(lclpath, &times);
          }
       return 1;
      }

   return (buf.st_mode & S_IFMT) == S_IFDIR;
}

/******************************************************************************/
/*                                M k p a t h                                 */
/******************************************************************************/

int XrdOlbServer::Mkpath(char *local_path, mode_t mode)
{
    char *next_path;
    struct stat buf;
    int i;

// Typically, the path exists, so check if so
//
   if (!stat(local_path, &buf)) return 0;

// Trim off the trailing slashes so we can have predictable behaviour
//
   i = strlen(local_path);
   while(i && local_path[--i] == '/') local_path[i] = '\0';
   if (!i) return ENOENT;

// Start creating directories starting with the root
//
   next_path = local_path;
   while((next_path = index(next_path+1, int('/'))))
        {*next_path = '\0';
         if (mkdir(local_path, mode) && errno != EEXIST) return errno;
         *next_path = '/';
        }

// Create last component and return
//
   if (mkdir(local_path, mode) && errno != EEXIST) return errno;
   return 0;
}

/******************************************************************************/
/*                              p r e p S c a n                               */
/******************************************************************************/
  
char *XrdOlbServer::prepScan(char **Line,XrdOlbPrepArgs *pargs,const char *Etxt)
{
   char *np, *tp = *Line;

   while(*tp && *tp == ' ') tp++;
   if (!(*tp))
      {Say.Emsg("Server", Etxt, "from", Name());
       delete pargs;
       return 0;
      }

   np = tp+1;
   while(*np && *np != ' ') np++;
   if (*np) *np++ = '\0';
   *Line = np;
   return tp;
}

/******************************************************************************/
/*                               R e c e i v e                                */
/******************************************************************************/
  
char *XrdOlbServer::Receive(char *idbuff, int blen)
{
   EPNAME("Receive")
   char *lp, *tp;

   if ((lp=Link->GetLine()) && *lp)
      {if (Trace.What & TRACE_Debug
       &&  strcmp("1@0 ping", lp) && strcmp("1@0 pong", lp))
           TRACEX("from " <<myNick <<": " <<lp);
       isActive = 1;
       if ((tp=Link->GetToken()))
          {strncpy(idbuff, tp, blen-2); idbuff[blen-1] = '\0';
           return Link->GetToken();
          }
      } else DEBUG("Null line from " <<myNick);
   return 0;
}
 
/******************************************************************************/
/*                               R e i s s u e                                */
/******************************************************************************/

int XrdOlbServer::Reissue(char *rid, const char *op,   char *arg1,
                                           char *path, char *arg3)
{
   XrdOlbPInfo pinfo;
   SMask_t amask;
   struct iovec iod[12];
   char newmid[32];
   int iocnt;

// Check if we can really reissue the command
//
   if (!(iod[0].iov_len = Config.GenMsgID(rid, newmid, sizeof(newmid))))
      {Say.Emsg("Server", "msg TTL exceeded for", op, path);
       return 0;
      }
   iod[0].iov_base = newmid;
   iocnt = 1;
  
// Find all the servers that might be able to do somthing on this path
//
   if (!Cache.Paths.Find(path, pinfo)
   || (amask = pinfo.rwvec | pinfo.rovec) == 0)
      {Say.Emsg("Server",op,"aborted; no servers handling path",path);
       return 0;
      }

// Construct the message
//
       iod[iocnt].iov_base = (char *)op;   iod[iocnt++].iov_len = strlen(op);
   if (arg1)
      {iod[iocnt].iov_base = arg1;         iod[iocnt++].iov_len = strlen(arg1);
       iod[iocnt].iov_base = (char *)" ";  iod[iocnt++].iov_len = 1;
      }
       iod[iocnt].iov_base = path;         iod[iocnt++].iov_len = strlen(path);
   if (arg3)
      {iod[iocnt].iov_base = (char *)" ";  iod[iocnt++].iov_len = 1;
       iod[iocnt].iov_base = arg3;         iod[iocnt++].iov_len = strlen(arg3);
       iod[iocnt].iov_base = (char *)" ";  iod[iocnt++].iov_len = 1;
      }
       iod[iocnt].iov_base = (char *)"\n"; iod[iocnt++].iov_len = 1;

// Now send off the message
//
   Manager.Broadcast(amask, iod, iocnt);
   return 0;
}
