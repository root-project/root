/******************************************************************************/
/*                                                                            */
/*                     X r d C m s P r o t o c o l . c c                      */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <netinet/in.h>
#include <sys/param.h>

#include "XProtocol/YProtocol.hh"

#include "Xrd/XrdInet.hh"
#include "Xrd/XrdLink.hh"

#include "XrdCms/XrdCmsCache.hh"
#include "XrdCms/XrdCmsCluster.hh"
#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsJob.hh"
#include "XrdCms/XrdCmsLogin.hh"
#include "XrdCms/XrdCmsManager.hh"
#include "XrdCms/XrdCmsManTree.hh"
#include "XrdCms/XrdCmsMeter.hh"
#include "XrdCms/XrdCmsProtocol.hh"
#include "XrdCms/XrdCmsRouting.hh"
#include "XrdCms/XrdCmsRTable.hh"
#include "XrdCms/XrdCmsState.hh"
#include "XrdCms/XrdCmsTrace.hh"

#include "XrdNet/XrdNetDNS.hh"

#include "XrdOuc/XrdOucCRC.hh"
#include "XrdOuc/XrdOucPup.hh"
#include "XrdOuc/XrdOucTokenizer.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysTimer.hh"

using namespace XrdCms;
  
/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

       XrdSysMutex      XrdCmsProtocol::ProtMutex;
       XrdCmsProtocol  *XrdCmsProtocol::ProtStack = 0;

       int              XrdCmsProtocol::readWait  = 1000;

       XrdCmsParser     XrdCmsProtocol::ProtArgs;

/******************************************************************************/
/*                       P r o t o c o l   L o a d e r                        */
/*                        X r d g e t P r o t o c o l                         */
/******************************************************************************/
  
// This protocol can live in a shared library. It can also be statically linked
// to provide a default protocol (which, for cms protocol we do). The interface
// below is used by Xrd to obtain a copy of the protocol object that can be
// used to decide whether or not a link is talking our particular protocol.
// Phase 1 initialization occured on the call to XrdgetProtocolPort(). At this
// point a network interface is defined and we can complete initialization.
//

extern "C"
{
XrdProtocol *XrdgetProtocol(const char *pname, char *parms,
                            XrdProtocol_Config *pi)
{
// If we failed in Phase 1 initialization, immediately fail Phase 2.
//
   if (Config.doWait < 0) return (XrdProtocol *)0;

// Initialize the network interface and get the actual port number assigned
//
   Config.PortTCP = pi->NetTCP->Port();
   Config.NetTCP  = pi->NetTCP;

// If we have a connection allow list, add it to the network object. Note that
// we clear the address because the object is lost in the add process.
//
   if (Config.Police) {pi->NetTCP->Secure(Config.Police); Config.Police = 0;}

// Complete initialization and upon success return a protocol object
//
   if (Config.Configure2()) return (XrdProtocol *)0;

// Return a new instance of this object
//
   return (XrdProtocol *)new XrdCmsProtocol();
}
}

/******************************************************************************/
/*           P r o t o c o l   P o r t   D e t e r m i n a t i o n            */
/*                    X r d g e t P r o t o c o l P o r t                     */
/******************************************************************************/
  
// Because the dcm port numbers are determined dynamically based on the role the
// dcm plays, we need to process the configration file and return the right
// port number if it differs from the one provided by the protocol driver. Only
// one port instance of the cmsd protocol is allowed.
//

extern "C"
{
int XrdgetProtocolPort(const char *pname, char *parms,
                       XrdProtocol_Config *pi)
{
   static int thePort = -1;
   char *cfn = pi->ConfigFN, buff[128];

// Check if we have been here before
//
   if (thePort >= 0)
      {if (pi->Port && pi->Port != thePort)
          {sprintf(buff, "%d disallowed; only using port %d",pi->Port,thePort);
           Say.Emsg("Config", "Alternate port", buff);
          }
       return thePort;
      }

// Initialize the error message handler and some default values
//
   Say.logger(pi->eDest->logger(0));
   Config.myName    = strdup(pi->myName);
   Config.PortTCP   = (pi->Port < 0 ? 0 : pi->Port);
   Config.myInsName = strdup(pi->myInst);
   Config.myProg    = strdup(pi->myProg);
   Sched            = pi->Sched;
   if (pi->DebugON) Trace.What = TRACE_ALL;
   memcpy(&Config.myAddr, pi->myAddr, sizeof(struct sockaddr));

// The only parameter we accept is the name of an alternate config file
//
   if (parms) 
      {while(*parms == ' ') parms++;
       if (*parms) 
          {char *pp = parms;
           while(*parms != ' ' && *parms) parms++;
           cfn = pp;
          }
      }

// Put up the banner
//
   Say.Say("Copr.  2007 Stanford University/SLAC cmsd.");

// Indicate failure if static init fails
//
   if (cfn) cfn = strdup(cfn);
   if (Config.Configure1(pi->argc, pi->argv, cfn))
      {Config.doWait = -1; return 0;}

// Return the port number to be used
//
   thePort = Config.PortTCP;
   return thePort;
}
}

/******************************************************************************/
/*                               E x e c u t e                                */
/******************************************************************************/
  
int XrdCmsProtocol::Execute(XrdCmsRRData &Arg)
{
   EPNAME("Execute");
   static kXR_unt32 theDelay = htonl(Config.SUPDelay);
   XrdCmsRouter::NodeMethod_t Method;
   const char *etxt;

// Check if we can continue
//
   if (CmsState.Suspended && Arg.Routing & XrdCmsRouting::Delayable)
      {Reply_Delay(Arg, theDelay); return 0;}

// Validate request code and execute the request. If successful, forward the
// request to subscribers of this node if the request is forwardable.
//
   if (!(Method = Router.getMethod(Arg.Request.rrCode)))
      Say.Emsg("Protocol", "invalid request code from", myNode->Ident);
      else if ((etxt = (myNode->*Method)(Arg)))
              if (*etxt == '!')
                 {DEBUGR(etxt+1 <<" delayed " <<Arg.waitVal <<" seconds");
                  return -EINPROGRESS;
                 } else if (*etxt == '.') return -ECONNABORTED;
                           else Reply_Error(Arg, kYR_EINVAL, etxt);
              else if (Arg.Routing & XrdCmsRouting::Forward && Cluster.NodeCnt
                   &&  !(Arg.Request.modifier & kYR_dnf)) Reissue(Arg);
   return 0;
}

/******************************************************************************/
/*                                 M a t c h                                  */
/******************************************************************************/

XrdProtocol *XrdCmsProtocol::Match(XrdLink *lp)
{
CmsRRHdr          Hdr;
int               dlen;

// Peek at the first few bytes of data (shouldb be all zeroes)
//
   if ((dlen = lp->Peek((char *)&Hdr,sizeof(Hdr),readWait)) != sizeof(Hdr))
      {if (dlen <= 0) lp->setEtext("login not received");
       return (XrdProtocol *)0;
      }

// Verify that this is our protocol and whether a version1 client is here
//
   if (Hdr.streamid || Hdr.rrCode != kYR_login)
      {if (!strncmp((char *)&Hdr, "login ", 6))
          lp->setEtext("protocol version 1 unsupported");
       return (XrdProtocol *)0;
      }

// Return the protocol object
//
   return (XrdProtocol *)XrdCmsProtocol::Alloc();
}

/******************************************************************************/
/*                                P a n d e r                                 */
/******************************************************************************/
  
// Pander() handles all outgoing connections to a manager/supervisor

void XrdCmsProtocol::Pander(const char *manager, int mport)
{
   EPNAME("Pander");

   CmsLoginData Data, loginData;
   unsigned int Mode, Role = 0;
   int Lvl=0, Netopts=0, waits=6, tries=6, fails=0, xport=mport;
   int rc, fsUtil, KickedOut, myNID = ManTree.Register();
   int chk4Suspend = XrdCmsState::All_Suspend, TimeOut = Config.AskPing*1000;
   char manbuff[256];
   const char *Reason = 0, *manp = manager;
   const int manblen = sizeof(manbuff);

// Do some debugging
//
   DEBUG(myRole <<" services to " <<manager <<':' <<mport);

// Prefill the login data
//
   loginData.SID   = (kXR_char *)Config.mySID;
   loginData.Paths = (kXR_char *)Config.myPaths;
   loginData.sPort = Config.PortTCP;
   loginData.fsNum = Meter.numFS();
   loginData.tSpace= Meter.TotalSpace(loginData.mSpace);

   loginData.Version = kYR_Version; // These to keep compiler happy
   loginData.HoldTime= static_cast<int>(getpid());
   loginData.Mode    = 0;
   loginData.Size    = 0;

// Establish request routing based on who we are
//
   if (Config.asManager()) Routing = (Config.asServer() ? &supVOps : &manVOps);
      else                 Routing = (Config.asPeer()   ? &supVOps : &srvVOps);

// Compute the Manager's status (this never changes for managers/supervisors)
//
   if (Config.asPeer())                Role  = CmsLoginData::kYR_peer;
      else if (Config.asManager())     Role  = CmsLoginData::kYR_manager;
              else                     Role  = CmsLoginData::kYR_server;
   if (Config.asProxy())               Role |= CmsLoginData::kYR_proxy;

// If we are a simple server, permanently add the nostage option if we are
// not able to stage any files.
//
   if (Role == CmsLoginData::kYR_server)
      {if (!Config.DiskSS) Role |=  CmsLoginData::kYR_nostage;}
      else chk4Suspend = XrdCmsState::FES_Suspend;

// Keep connecting to our manager. If suspended, wait for a resumption first
//
   do {if (Config.doWait && chk4Suspend)
          while(CmsState.Suspended & chk4Suspend)
               {if (!waits--)
                   {Say.Emsg("Pander", "Suspend state still active."); waits=6;}
                XrdSysTimer::Snooze(12);
               }

       if (!ManTree.Trying(myNID, Lvl) && Lvl)
          {DEBUG("restarting at root node " <<manager <<':' <<mport);
           manp = manager; xport = mport; Lvl = 0;
          }

       DEBUG("trying to connect to lvl " <<Lvl <<' ' <<manp <<':' <<xport);

       if (!(Link = Config.NetTCP->Connect(manp, xport, Netopts)))
          {if (tries--) Netopts = XRDNET_NOEMSG;
              else {tries = 6; Netopts = 0;}
           if ((Lvl = myMans.Next(xport,manbuff,manblen)))
                   {XrdSysTimer::Snooze(3); manp = manbuff;}
              else {if (manp != manager) fails++;
                    XrdSysTimer::Snooze(6); manp = manager; xport = mport;
                   }
           continue;
          }
       Netopts = 0; tries = waits = 6;

       // Obtain a new node object for this connection
       //
       if (!(myNode = Manager.Add(Link, Lvl+1)))
          {Say.Emsg("Pander", "Unable to obtain node object.");
           Link->Close(); XrdSysTimer::Snooze(15); continue;
          }

      // Compute current login mode
      //
      Mode = Role
           | (CmsState.Suspended ? CmsLoginData::kYR_suspend : 0)
           | (CmsState.NoStaging ? CmsLoginData::kYR_nostage : 0);
       if (fails >= 6 && manp == manager) 
          {fails = 0; Mode |=    CmsLoginData::kYR_trying;}

       // Login this node with the correct state
       //
       loginData.fSpace= Meter.FreeSpace(fsUtil);
       loginData.fsUtil= static_cast<kXR_unt16>(fsUtil);
       KickedOut = 0; loginData.dPort = CmsState.Port();
       Data = loginData; Data.Mode = Mode;
       if (!(rc = XrdCmsLogin::Login(Link, Data, TimeOut)))
          {if(!ManTree.Connect(myNID, myNode)) KickedOut = 1;
             else {Say.Emsg("Protocol", "Logged into", Link->Name());
                   Reason = Dispatch(isUp, TimeOut, 2);
                   rc = 0;
                   loginData.fSpace= Meter.FreeSpace(fsUtil);
                   loginData.fsUtil= static_cast<kXR_unt16>(fsUtil);
                  }
          }

       // Remove manager from the config
       //
       Manager.Remove(myNode, (rc == kYR_redirect ? "redirected"
                                  : (Reason ? Reason : "lost connection")));
       ManTree.Disc(myNID);
       Link->Close();
       delete myNode; myNode = 0;

       // Check if we should process the redirection
       //
       if (rc == kYR_redirect)
          {struct sockaddr netaddr;
           XrdOucTokenizer hList((char *)Data.Paths);
           unsigned int ipaddr;
           char *hP;
           Link->Name(&netaddr);
           ipaddr = XrdNetDNS::IPAddr(&netaddr);
           myMans.Del(ipaddr);
           while((hP = hList.GetToken()))
                 myMans.Add(ipaddr, hP, Config.PortTCP, Lvl+1);
           free(Data.Paths);
          }

       // Cycle on to the next manager if we have one or snooze and try over
       //
       if (!KickedOut && (Lvl = myMans.Next(xport,manbuff,manblen)))
          {manp = manbuff; continue;}
       XrdSysTimer::Snooze(9); Lvl = 0;
       if (manp != manager) fails++;
       manp = manager; xport = mport;
      } while(1);
}
  
/******************************************************************************/
/*                               P r o c e s s                                */
/******************************************************************************/
  
// Process is called only when we get a new connection. We only return when
// the connection drops.
//
int XrdCmsProtocol::Process(XrdLink *lp)
{
   const char *Reason;
   Bearing     myWay;
   int         tOut;

// Now admit the login
//
   Link = lp;
   if ((Routing=Admit()))
      {loggedIn = 1;
       if (RSlot) {myWay = isLateral; tOut = -1;}
          else    {myWay = isDown;    tOut = Config.AskPing*1000;}
       myNode->UnLock();
       if ((Reason = Dispatch(myWay, tOut, 2))) lp->setEtext(Reason);
       myNode->Lock();
      }

// Serialize all activity on the link before we proceed
//
   lp->Serialize();

// Immediately terminate redirectors (they have an Rslot).
//
   if (RSlot)
      {RTable.Del(myNode); RSlot  = 0;
       myNode->UnLock(); delete myNode; myNode = 0;
       return -1;
      }

// We have a node that may or may not be in the cluster at this point, or may
// need to remain in the cluster as a shadow member. In any case, the node
// object lock will be released either by Remove() or the destructor.
//
   if (myNode)
      {myNode->isConn = 0;
       if (myNode->isBound) Cluster.Remove(0, myNode, !loggedIn);
          else if (myNode->isGone) delete myNode;
              else myNode->UnLock();
      }

// All done indicate the connection is dead
//
   return -1;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/

void XrdCmsProtocol::Recycle(XrdLink *lp, int consec, const char *reason)
{
   if (loggedIn) 
      if (reason) Say.Emsg("Protocol", lp->ID, "logged out;",   reason);
         else     Say.Emsg("Protocol", lp->ID, "logged out.");
      else     
      if (reason) Say.Emsg("Protocol", lp->ID, "login failed;", reason);

   ProtMutex.Lock();
   ProtLink  = ProtStack;
   ProtStack = this;
   ProtMutex.UnLock();
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                 A d m i t                                  */
/******************************************************************************/
  
XrdCmsRouting *XrdCmsProtocol::Admit()
{
   EPNAME("Admit");
   char         myBuff[1024];
   XrdCmsLogin  Source(myBuff, sizeof(myBuff));
   CmsLoginData Data;
   const char  *Reason;
   SMask_t      newmask, servset(0);
   int addedp = 0, Status = 0, isMan = 0, isPeer = 0, isProxy = 0, isServ = 0;
   int wasSuspended = 0;

// Establish outgoing mode
//
   Data.Mode = 0;
   if (Trace.What & TRACE_Debug) Data.Mode |= CmsLoginData::kYR_debug;
   if (CmsState.Suspended)      {Data.Mode |= CmsLoginData::kYR_suspend;
                                 wasSuspended = 1;
                                }
   Data.HoldTime = Config.LUPHold;

// Do the login and get the data
//
   if (!Source.Admit(Link, Data)) return 0;

// Handle Redirectors here (minimal stuff to do)
//
   if (Data.Mode & CmsLoginData::kYR_director) 
      {Link->setID("redirector", Data.HoldTime);
       return Admit_Redirector(wasSuspended);
      }

// Disallow subscriptions we are are configured as a solo manager
//
   if (Config.asSolo())
      return Login_Failed("configuration disallows subscribers");

// Determine the role of this incomming login.
//
        if ((isMan = Data.Mode & CmsLoginData::kYR_manager))
           {Status = CMS_isMan;
            if ((isPeer =  Data.Mode & CmsLoginData::kYR_peer))
               {Status |= CMS_isPeer;  myRole = "manager";}
               else                    myRole = "supervisor";
           }
   else if ((isServ =  Data.Mode & CmsLoginData::kYR_server))
            if ((isProxy=  Data.Mode & CmsLoginData::kYR_proxy))
               {Status = CMS_isProxy; myRole = "proxy_srvr";}
               else                   myRole = "server";
   else if ((isPeer =  Data.Mode & CmsLoginData::kYR_peer))
           {Status |= CMS_isPeer;
            myRole = (CmsLoginData::kYR_proxy ? "peer" : "peer_proxy");
           }
   else    return Login_Failed("invalid login role");

// Set the link identification
//
   Link->setID(myRole, Data.HoldTime);

// Make sure that our role is compatible with the incomming role
//
   Reason = 0;
        if (Config.asProxy()) {if (!isProxy || isPeer)
                                  Reason = "configuration only allows proxies";
                              }
   else if (isProxy)              Reason = "configuration disallows proxies";
   else if (Config.asServer() && isPeer)
                                  Reason = "configuration disallows peers";
   if (Reason) return Login_Failed(Reason);

// The server may specify nostage and suspend
//
   if (Data.Mode & CmsLoginData::kYR_nostage) Status |= CMS_noStage;
   if (Data.Mode & CmsLoginData::kYR_suspend) Status |= CMS_Suspend;

// The server may specify that it has been trying for a long time
//
   if (Data.Mode & CmsLoginData::kYR_trying)
      Say.Emsg("Protocol",Link->Name(),"has not yet found a cluster slot!");

// Add the node. The resulting node object will be locked and the caller will
// unlock it prior to dispatching.
//
   if (!(myNode = Cluster.Add(Link, Data.dPort, Status, Data.sPort,
                              (const char *)Data.SID)))
      return (XrdCmsRouting *)0;

// Record the status of the server's filesystem
//
   DEBUG(Link->Name() <<" TSpace=" <<Data.tSpace <<"GB NumFS=" <<Data.fsNum
                      <<" FSpace=" <<Data.fSpace <<"MB MinFR=" <<Data.mSpace
                      <<"MB Util=" <<Data.fsUtil);
   myNode->DiskTotal = Data.tSpace;
   myNode->DiskMinF  = Data.mSpace;
   myNode->DiskFree  = Data.fSpace;
   myNode->DiskNums  = Data.fsNum;
   myNode->DiskUtil  = Data.fsUtil;
   Meter.setVirtUpdt();

// Check for any configuration changes and then process all of the paths.
//
   if (Data.Paths && *Data.Paths)
      {XrdOucTokenizer thePaths((char *)Data.Paths);
       char *tp, *pp;
       ConfigCheck(Data.Paths);
       while((tp = thePaths.GetLine()))
            {DEBUG(Link->Name() <<" adding path: " <<tp);
             if (!(tp = thePaths.GetToken())
             ||  !(pp = thePaths.GetToken())) break;
             if (!(newmask = AddPath(myNode, tp, pp)))
                return Login_Failed("invalid exported path");
             servset |= newmask;
             addedp= 1;
            }
      }

// Check if we have any special paths. If none, then add the default path.
//
   if (!addedp) 
      {XrdCmsPInfo pinfo;
       ConfigCheck(0);
       pinfo.rovec = myNode->Mask();
       if (myNode->isPeer) pinfo.ssvec = myNode->Mask();
       servset = Cache.Paths.Insert("/", &pinfo);
       Say.Emsg("Protocol", myNode->Ident, "defaulted r /");
      }

// Set the reference counts for intersecting nodes to be the same.
// Additionally, indicate cache refresh will be needed because we have a new
// node that may have files the we already reported on.
//
   Cluster.ResetRef(servset);
   if (Config.asManager()) {Manager.Reset(); myNode->SyncSpace();}
   myNode->isDisable = 0;

// Document the login
//
   Say.Emsg("Protocol", myNode->Ident,
            (myNode->isSuspend ? "logged in suspended." : "logged in."));

// All done
//
   return &rspVOps;
}
  
/******************************************************************************/
/*                      A d m i t _ R e d i r e c t o r                       */
/******************************************************************************/
  
XrdCmsRouting *XrdCmsProtocol::Admit_Redirector(int wasSuspended)
{
   EPNAME("Admit_Redirector");
   static CmsStatusRequest newState 
                   = {{0, kYR_status, CmsStatusRequest::kYR_Resume, 0}};

// Indicate what role I have
//
   myRole = "redirector";

// Director logins have no additional parameters. We return with the node object
// locked to be consistent with the way server/suprvisors nodes are returned.
//
   myNode = new XrdCmsNode(Link); myNode->Lock();
   if (!(RSlot = RTable.Add(myNode)))
      {Say.Emsg("Protocol",myNode->Ident,"login failed; too many redirectors.");
       return 0;
      } else myNode->setSlot(RSlot);

// If we told the redirector we were suspended then we must check if that is no
// longer true and generate a reume event as the redirector may have missed it
//
   if (wasSuspended && !CmsState.Suspended)
      myNode->Send((char *)&newState, sizeof(newState));

// Login succeeded
//
   Say.Emsg("Protocol", myNode->Ident, "logged in.");
   DEBUG(myNode->Ident <<" assigned slot " <<RSlot);
   return &rdrVOps;
}

/******************************************************************************/
/*                               A d d P a t h                                */
/******************************************************************************/
  
SMask_t XrdCmsProtocol::AddPath(XrdCmsNode *nP,
                                const char *pType, const char *Path)
{
    XrdCmsPInfo pinfo;

// Process: addpath {r | w | rw}[s] path
//
   while(*pType)
        {     if ('r' == *pType) pinfo.rovec =               nP->Mask();
         else if ('w' == *pType) pinfo.rovec = pinfo.rwvec = nP->Mask();
         else if ('s' == *pType) pinfo.rovec = pinfo.ssvec = nP->Mask();
         else return 0;
         pType++;
        }

// Set node options
//
   nP->isRW = (pinfo.rwvec ? XrdCmsNode::allowsRW : 0) 
            | (pinfo.ssvec ? XrdCmsNode::allowsSS : 0);

// Add the path to the known path list
//
   return Cache.Paths.Insert(Path, &pinfo);
}

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/

XrdCmsProtocol *XrdCmsProtocol::Alloc(const char *theRole, 
                                      const char *theMan,
                                            int   thePort)
{
   XrdCmsProtocol *xp;

// Grab a protocol object and, if none, return a new one
//
   ProtMutex.Lock();
   if ((xp = ProtStack)) ProtStack = xp->ProtLink;
      else xp = new XrdCmsProtocol();
   ProtMutex.UnLock();

// Initialize the object if we actually got one
//
   if (!xp) Say.Emsg("Protocol","No more protocol objects.");
      else {xp->myRole    = theRole;
            xp->myMan     = theMan;
            xp->myManPort = thePort;
            xp->loggedIn  = 0;
           }

// All done
//
   return xp;
}

/******************************************************************************/
/*                           C o n f i g C h e c k                            */
/******************************************************************************/
  
void XrdCmsProtocol::ConfigCheck(unsigned char *theConfig)
{
  unsigned int ConfigID;
  int tmp;

// Compute the new configuration ID
//
   if (!theConfig) ConfigID = 1;
      else ConfigID = XrdOucCRC::CRC32(theConfig, strlen((char *)theConfig));

// If the configuration chaged or a new node, then we must bounce this node
//
   if (ConfigID != myNode->ConfigID)
      {if (myNode->ConfigID) Say.Emsg("Protocol",Link->Name(),"reconfigured.");
       Cache.Paths.Remove(myNode->Mask());
       Cache.Bounce(myNode->Mask(), myNode->ID(tmp));
       myNode->ConfigID = ConfigID;
      }
}

/******************************************************************************/
/*                              D i s p a t c h                               */
/******************************************************************************/

// Dispatch is provided with three key pieces of information:
// 1) The connection bearing (isUp, isDown, isLateral) the determines how
//    timeouts are to be handled.
// 2) The maximum amount to wait for data to arrive.
// 3) The number of successive timeouts we can have before we give up.
  
const char *XrdCmsProtocol::Dispatch(Bearing cDir, int maxWait, int maxTries)
{
   EPNAME("Dispatch");
   static const int ReqSize = sizeof(CmsRRHdr);
   static CmsPingRequest Ping = {{0, kYR_ping,  0, 0}};
   XrdCmsRRData *Data = XrdCmsRRData::Objectify();
   XrdCmsJob  *jp;
   const char *toRC = (cDir == isUp ? "manager not active"
                                    : "server not responding");
   const char *myArgs, *myArgt;
   char        buff[8];
   int         rc, toLeft = maxTries;

// Dispatch runs with the current thread bound to the link.
//
   Link->Bind(XrdSysThread::ID());

// Read in the request header
//
do{if ((rc = Link->RecvAll((char *)&Data->Request, ReqSize, maxWait)) < 0)
      {if (rc != -ETIMEDOUT) return "request read failed";
       if (!toLeft--) return toRC;
       if (cDir == isDown && Link->Send((char *)&Ping, sizeof(Ping)) < 0)
          return "server unreachable";
       continue;
      }

// Decode the length and get the rest of the data
//
   toLeft = maxTries;
   Data->Dlen = static_cast<int>(ntohs(Data->Request.datalen));
   if ((QTRACE(Debug))
   && Data->Request.rrCode != kYR_ping && Data->Request.rrCode != kYR_pong)
      DEBUG(myNode->Ident <<" for " <<Router.getName(Data->Request.rrCode)
                          <<" dlen=" <<Data->Dlen);
   if (!(Data->Dlen)) {myArgs = myArgt = 0;}
      else {if (Data->Dlen > maxReqSize)
               {Say.Emsg("Protocol","Request args too long from",Link->Name());
                return "protocol error";
               }
            if ((!Data->Buff || Data->Blen < Data->Dlen)
            &&  !Data->getBuff(Data->Dlen))
               {Say.Emsg("Protocol", "No buffers to serve", Link->Name());
                return "insufficient buffers";
               }
            if ((rc = Link->RecvAll(Data->Buff, Data->Dlen, maxWait)) < 0)
               return (rc == -ETIMEDOUT ? "read timed out" : "read failed");
            myArgs = Data->Buff; myArgt = Data->Buff + Data->Dlen;
           }

// Check if request is actually valid
//
   if (!(Data->Routing = Routing->getRoute(int(Data->Request.rrCode))))
      {sprintf(buff, "%d", Data->Request.rrCode);
       Say.Emsg("Protocol",Link->Name(),"sent an invalid request -", buff);
       continue;
      }

// Parse the arguments (we do this in the main thread to avoid overruns)
//
   if (!(Data->Routing & XrdCmsRouting::noArgs))
      {if (Data->Request.modifier & kYR_raw)
          {Data->Path = Data->Buff; Data->PathLen = Data->Dlen;}
          else if (!myArgs
               || !ProtArgs.Parse(int(Data->Request.rrCode),myArgs,myArgt,Data))
                  {Reply_Error(*Data, kYR_EINVAL, "badly formed request");
                   continue;
                  }
      }

// Insert correct identification
//
   if (!(Data->Ident) || !(*Data->Ident)) Data->Ident = myNode->Ident;

// Schedule this request if async. Otherwise, do this inline. Note that
// synchrnous requests are allowed to return status changes (e.g., redirect)
//
   if (Data->Routing & XrdCmsRouting::isSync)
      {if ((rc = Execute(*Data)) && rc == -ECONNABORTED) return "redirected";}
      else if ((jp = XrdCmsJob::Alloc(this, Data)))
              {Sched->Schedule((XrdJob *)jp);
               Data = XrdCmsRRData::Objectify();
              }
              else Say.Emsg("Protocol", "No jobs to serve", Link->Name());
  } while(1);

// We should never get here
//
   return "logic error";
}

/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/
  
// Determine how we should proceed here
//
void XrdCmsProtocol::DoIt()
{

// If we have a role, then we should simply pander it
//
   if (myRole) Pander(myMan, myManPort);
}

/******************************************************************************/
/*                          L o g i n _ F a i l e d                           */
/******************************************************************************/
  
XrdCmsRouting *XrdCmsProtocol::Login_Failed(const char *reason)
{
   Link->setEtext(reason);
   return (XrdCmsRouting *)0;
}

/******************************************************************************/
/*                               R e i s s u e                                */
/******************************************************************************/

void XrdCmsProtocol::Reissue(XrdCmsRRData &Data)
{
   EPNAME("Resisue");
   XrdCmsPInfo pinfo;
   SMask_t amask;
   struct iovec ioB[2] = {{(char *)&Data.Request, sizeof(Data.Request)},
                          {         Data.Buff,    Data.Dlen}
                         };

// Check if we can really reissue the command
//
   if (!((Data.Request.modifier += kYR_hopincr) & kYR_hopcount))
      {Say.Emsg("Job", Router.getName(Data.Request.rrCode),
                       "msg TTL exceeded for", Data.Path);
       return;
      }

// We do not support 2way re-issued messages
//
   Data.Request.streamid = 0;
  
// Find all the nodes that might be able to do somthing on this path
//
   if (!Cache.Paths.Find(Data.Path, pinfo)
   || (amask = pinfo.rwvec | pinfo.rovec) == 0)
      {Say.Emsg("Job", Router.getName(Data.Request.rrCode),
                       "aborted; no servers handling", Data.Path);
       return;
      }

// Do some debugging
//
   DEBUG("FWD " <<Router.getName(Data.Request.rrCode) <<' ' <<Data.Path);

// Now send off the message to all the nodes
//
   Cluster.Broadcast(amask, ioB, 2, sizeof(Data.Request)+Data.Dlen);
}
  
/******************************************************************************/
/*                           R e p l y _ D e l a y                            */
/******************************************************************************/
  
void XrdCmsProtocol::Reply_Delay(XrdCmsRRData &Data, kXR_unt32 theDelay)
{
     EPNAME("Reply_Delay");
     const char *act;

     if (Data.Request.streamid && (Data.Routing & XrdCmsRouting::Repliable))
        {CmsResponse Resp = {{Data.Request.streamid, kYR_wait, 0,
                               htons(sizeof(kXR_unt32))}, theDelay};
         act = " sent";
         Link->Send((char *)&Resp, sizeof(Resp));
        } else act = " skip";

     DEBUG(myNode->Ident <<act <<" delay " <<ntohl(theDelay));
}

/******************************************************************************/
/*                           R e p l y _ E r r o r                            */
/******************************************************************************/
  
void XrdCmsProtocol::Reply_Error(XrdCmsRRData &Data, int ecode, const char *etext)
{
     EPNAME("Reply_Error");
     const char *act;
     int n = strlen(etext)+1;

     if (Data.Request.streamid && (Data.Routing & XrdCmsRouting::Repliable))
        {CmsResponse Resp = {{Data.Request.streamid, kYR_error, 0,
                              htons(sizeof(kXR_unt32)+n)},
                             htonl(static_cast<unsigned int>(ecode))};
         struct iovec ioV[2] = {{(char *)&Resp, sizeof(Resp)},
                                {(char *)etext, n}};
         act = " sent";
         Link->Send(ioV, 2);
        } else act = " skip";

     DEBUG(myNode->Ident <<act <<" err " <<ecode  <<' ' <<etext);
}
