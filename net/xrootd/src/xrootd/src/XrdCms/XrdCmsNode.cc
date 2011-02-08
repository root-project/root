/***********************************************************************************************/
/*                                                                            */
/*                         X r d C m s N o d e . c c                          */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

// Original Version: 1.52 2007/08/28 01:32:15 abh

const char *XrdCmsNodeCVSID = "$Id$";
  
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <time.h>
#include <netinet/in.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Xrd/XrdJob.hh"
#include "Xrd/XrdLink.hh"

#include "XProtocol/YProtocol.hh"

#include "XrdCms/XrdCmsCache.hh"
#include "XrdCms/XrdCmsCluster.hh"
#include "XrdCms/XrdCmsConfig.hh"
#include "XrdCms/XrdCmsManager.hh"
#include "XrdCms/XrdCmsManList.hh"
#include "XrdCms/XrdCmsManTree.hh"
#include "XrdCms/XrdCmsMeter.hh"
#include "XrdCms/XrdCmsPList.hh"
#include "XrdCms/XrdCmsPrepare.hh"
#include "XrdCms/XrdCmsRRData.hh"
#include "XrdCms/XrdCmsNode.hh"
#include "XrdCms/XrdCmsSelect.hh"
#include "XrdCms/XrdCmsState.hh"
#include "XrdCms/XrdCmsTrace.hh"
#include "XrdCms/XrdCmsXmi.hh"

#include "XrdNet/XrdNetDNS.hh"

#include "XrdOss/XrdOss.hh"

#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucPup.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdOuc/XrdOucUtils.hh"

#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdCms;

/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/

XrdSysMutex XrdCmsNode::mlMutex;

int         XrdCmsNode::LastFree = 0;
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCmsNode::XrdCmsNode(XrdLink *lnkp, int port,
                       const char *nid,  int lvl, int id)
{
    static XrdSysMutex   iMutex;
    static const SMask_t smask_1(1);
    static int           iNum = 1;

    Link     =  lnkp;
    IPAddr   =  0;
    NodeMask =  (id < 0 ? 0 : smask_1 << id);
    NodeID   = id;
    isDisable=  0;
    isNoStage=  0;
    isOffline=  (lnkp == 0);
    isSuspend=  0;
    isBound  =  0;
    isConn   =  0;
    isGone   =  0;
    isPerm   =  0;
    isMan    =  0;
    isKnown  =  0;
    isPeer   =  0;
    isProxy  =  0;
    myCost   =  0;
    myLoad   =  0;
    myMass   =  0;
    myCNUM   = -3;
    DiskTotal=  0;
    DiskFree =  0;
    DiskMinF =  0;
    DiskNums =  0;
    DiskUtil =  0;
    Next     =  0;
    RefA     =  0;
    RefTotA  =  0;
    RefR     =  0;
    RefTotR  =  0;
    logload  =  Config.LogPerf;
    DropTime =  0;
    DropJob  =  0;
    myName   =  0;
    myNlen   =  0;
    Ident    =  0;
    Port     =  0;
    myNID    = strdup(nid ? nid : "?");
    if ((myCID = index(myNID, ' '))) myCID++;
       else myCID = myNID;
    myLevel  = lvl;
    ConfigID =  0;

// setName() will set Ident, IPAddr, IPV6, myName, myNlen, & Port!
//
   setName(lnkp, (nid ? port : 0));

   iMutex.Lock();
   Instance =  iNum++;
   iMutex.UnLock();
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdCmsNode::~XrdCmsNode()
{
   isOffline = 1;
   if (isLocked) UnLock();

// Delete other appendages
//
   if (Ident) free(Ident);
   if (myNID) free(myNID);
   if (myName)free(myName);
}

/******************************************************************************/
/*                               s e t N a m e                                */
/******************************************************************************/
  
void XrdCmsNode::setName(XrdLink *lnkp, int port)
{
   struct sockaddr netaddr;
   char *bp, buff[512];
   const char *hname = lnkp->Host();
   unsigned int hAddr;

// Get our address (the long way)
//
   lnkp->Name(&netaddr);
   hAddr= XrdNetDNS::IPAddr(&netaddr);

// Check if this is a duplicate
//
   if (myName)
      {if (!strcmp(myName, hname) && port == Port && hAddr == IPAddr) return;
          else free(myName);
      }

// Construct our identification
//
   IPAddr = hAddr;
   myName = strdup(hname);
   myNlen = strlen(hname)+1;
   Port = port;

   if (!port) strcpy(buff, lnkp->ID);
      else    sprintf(buff, "%s:%d", lnkp->ID, port);
   if (Ident) free(Ident);
   Ident = strdup(buff);

   strcpy(IPV6, "[::");
   bp = IPV6+3;
   bp += XrdNetDNS::IP2String(IPAddr, 0, bp, 24); // We're cheating
   *bp++ = ']';
   if (Port) {*bp++ = ':'; bp += sprintf(bp, "%d", Port);}
   IPV6Len = bp - IPV6;
}

/******************************************************************************/
/*                                  D i s c                                   */
/******************************************************************************/

void XrdCmsNode::Disc(const char *reason, int needLock)
{

// Lock the object of not yet locked
//
   if (needLock) myMutex.Lock();
   isOffline = 1;

// If we are still connected, initiate a teardown
//
   if (isConn)
      {Link->setEtext(reason);
       Link->Close(1);
       isConn = 0;
      }

// Unlock ourselves if we locked ourselves
//
   if (needLock) myMutex.UnLock();
}
  
/******************************************************************************/
/*                              d o _ A v a i l                               */
/******************************************************************************/
  
// Node responses to space usage requests from a manager are localized to the 
// cell and need not be propopagated in any direction.
//
const char *XrdCmsNode::do_Avail(XrdCmsRRData &Arg)
{
   EPNAME("do_Avail")

// Process: avail <fsdsk> <util>
//
   DiskFree = Arg.dskFree;
   DiskUtil = static_cast<int>(Arg.dskUtil);

// Do some debugging
//
   DEBUGR(DiskFree <<"MB free; " <<DiskUtil <<"% util");
   return 0;
}

/******************************************************************************/
/*                              d o _ C h m o d                               */
/******************************************************************************/
  
// Chmod requests are forwarded to all subscribers subject to an Xmi callout.
//
const char *XrdCmsNode::do_Chmod(XrdCmsRRData &Arg)
{
   EPNAME("do_Chmod")
   mode_t mode = 0;
   int rc;

// Do some debugging
//
   DEBUGR("mode " <<Arg.Mode <<' ' <<Arg.Path);

// If we have an Xmi then call it
//
   if (Xmi_Chmod)
      {XrdCmsReq Req(this, Arg.Request.streamid);
       if (!getMode(Arg.Mode, mode)) return "invalid mode";
          else if (Xmi_Chmod->Chmod(&Req, mode, Arg.Path, Arg.Opaque)) return 0;
      }

// We are don here if we have no data; otherwise convert the mode if we
// haven't done so already.
//
   if (!Config.DiskOK) return 0;
   if (!mode && !getMode(Arg.Mode, mode)) return "invalid mode";

// Attempt to change the mode either via call-out or the oss plug-in
//
   if (Config.ProgCH) rc = fsExec(Config.ProgCH, Arg.Mode, Arg.Path);
      else rc = Config.ossFS->Chmod(Arg.Path, mode);

// Return appropriate result
//
   return (rc ? fsFail(Arg.Ident, "chmod", Arg.Path, rc) : 0);
}

/******************************************************************************/
/*                               d o _ D i s c                                */
/******************************************************************************/

// When a manager receives a disc response from a node it sends a disc request
// and then closes the connection.
// When a node    receives a disc request it simply closes the connection.

const char *XrdCmsNode::do_Disc(XrdCmsRRData &Arg)
{

// Indicate we have received a disconnect
//
   Say.Emsg("Node", Link->Name(), "requested a disconnect");

// If we must send a disc request, do so now
//
   if (Config.asManager()) Link->Send((char *)&Arg.Request,sizeof(Arg.Request));

// Close the link and return an error
//
   isOffline = 1;
   Link->Close(1);
   return ".";   // Signal disconnect
}

/******************************************************************************/
/*                               d o _ G o n e                                */
/******************************************************************************/

// When a manager receives a gone request it is propogated if we are subscribed
// and we have not sent a gone request in the immediate past.
//
const char *XrdCmsNode::do_Gone(XrdCmsRRData &Arg)
{
   EPNAME("do_Gone")
   int newgone;

// Do some debugging
//
   TRACER(Files,Arg.Path);

// Update path information and delete this from the prep queue if we are a
// staging node. We can also be called via the admin end-point interface
// In this case, we have no cache and simply forward up the request.
//
   if (Config.asManager())
      {XrdCmsSelect Sel(XrdCmsSelect::Advisory, Arg.Path, Arg.PathLen-1);
       newgone = Cache.DelFile(Sel, NodeMask);
      } else {
       newgone = 1;
       if (Config.DiskSS) PrepQ.Gone(Arg.Path);
      }

// If we have no managers and we still have the file or never had it, return
//
   if (!Manager.Present() || !newgone) return 0;

// Back-propogate the gone to all of our managers
//
   Manager.Inform(Arg.Request, Arg.Buff, Arg.Dlen);

// All done
//
   return 0;
}

/******************************************************************************/
/*                               d o _ H a v e                                */
/******************************************************************************/
  
// When a manager receives a have request it is propogated if we are subscribed
// and we have not sent a have request in the immediate past.
//
const char *XrdCmsNode::do_Have(XrdCmsRRData &Arg)
{
   EPNAME("do_Have")
   XrdCmsPInfo  pinfo;
   int isnew, Opts;

// Do some debugging
//
   TRACER(Files, (Arg.Request.modifier&CmsHaveRequest::Pending ? "P ":"") 
                 <<Arg.Path);

// Find if we can handle the file in r/w mode and if staging is present
//
   Opts = (Cache.Paths.Find(Arg.Path, pinfo) && (pinfo.rwvec & NodeMask)
        ? XrdCmsSelect::Write : 0);
   if (Arg.Request.modifier & CmsHaveRequest::Pending)
      Opts |= XrdCmsSelect::Pending;

// Update path information
//
   if (!Config.asManager()) isnew = 1;
      else {XrdCmsSelect Sel(XrdCmsSelect::Advisory|Opts,Arg.Path,Arg.PathLen-1);
            Sel.Path.Hash = Arg.Request.streamid;
            isnew = Cache.AddFile(Sel, NodeMask);
           }

// Return if we have no managers or we already informed the managers
//
   if (!Manager.Present() || !isnew) return 0;

// Back-propogate the have to all of our managers
//
   Manager.Inform(Arg.Request, Arg.Buff, Arg.Dlen);

// All done
//
   return 0;
}
  
/******************************************************************************/
/*                               d o _ L o a d                                */
/******************************************************************************/
  
// Responses to usage requests are local to the cell and never propagated.
//
const char *XrdCmsNode::do_Load(XrdCmsRRData &Arg)
{
   EPNAME("do_Load")
   int temp, pcpu, pnet, pxeq, pmem, ppag, pdsk;

// Process: load <cpu> <io> <load> <mem> <pag> <util> <rsvd> <dskFree>
//               0     1    2      3     4     5      6
   pcpu = static_cast<int>(Arg.Opaque[CmsLoadRequest::cpuLoad]);
   pnet = static_cast<int>(Arg.Opaque[CmsLoadRequest::netLoad]);
   pxeq = static_cast<int>(Arg.Opaque[CmsLoadRequest::xeqLoad]);
   pmem = static_cast<int>(Arg.Opaque[CmsLoadRequest::memLoad]);
   ppag = static_cast<int>(Arg.Opaque[CmsLoadRequest::pagLoad]);
   pdsk = static_cast<int>(Arg.Opaque[CmsLoadRequest::dskLoad]);

// Compute actual load value
//
   myLoad = Meter.calcLoad(pcpu, pnet, pxeq, pmem, ppag);
   myMass = Meter.calcLoad(myLoad, pdsk);
   DiskFree = Arg.dskFree;
   DiskUtil = pdsk;

// Do some debugging
//
   DEBUGR("cpu=" <<pcpu <<" net=" <<pnet <<" xeq=" <<pxeq
       <<" mem=" <<pmem <<" pag=" <<ppag <<" dsk=" <<pdsk
       <<"% " <<DiskFree <<"MB load=" <<myLoad <<" mass=" <<myMass);

// If we are also a manager then use this load figure to come up with
// an overall load to report when asked. If we get free space, then we
// must report that now so that we can be selected for allocation.
//
   if (Config.asManager())
      {Meter.Record(pcpu, pnet, pxeq, pmem, ppag, pdsk);
       if (isRW && DiskFree != LastFree)
          {mlMutex.Lock();
           temp = LastFree; LastFree = DiskFree; Meter.setVirtUpdt();
           if (!temp && DiskFree >= Config.DiskMin) do_Space(Arg);
           mlMutex.UnLock();
          }
      }

// Report new load if need be
//
   if (Config.LogPerf && !logload)
      {char buff[1024];
       snprintf(buff, sizeof(buff)-1,
               "load=%d; cpu=%d net=%d inq=%d mem=%d pag=%d dsk=%d utl=%d",
               myLoad, pcpu, pnet, pxeq, pmem, ppag, Arg.dskFree, pdsk);
       Say.Emsg("Node", Name(), buff);
       logload = Config.LogPerf;
      } else logload--;

// Return as if we had gotten a pong
//
   return do_Pong(Arg);
}

  
/******************************************************************************/
/*                             d o _ L o c a t e                              */
/******************************************************************************/

const char *XrdCmsNode::do_Locate(XrdCmsRRData &Arg)
{
   EPNAME("do_Locate";)
   XrdCmsRRQInfo reqInfo(Instance, RSlot, Arg.Request.streamid);
   XrdCmsSelect    Sel(0, Arg.Path, Arg.PathLen-1);
   XrdCmsSelected *sP = 0;
   struct {kXR_unt32 Val; 
           char outbuff[CmsLocateRequest::RILen*STMax];} Resp;
   struct iovec ioV[2] = {{(char *)&Arg.Request, sizeof(Arg.Request)},
                          {(char *)&Resp,        0}};
   const char *Why;
   char theopts[8], *toP = theopts;
   int rc, bytes;

// Do a callout to the external manager if we have one
//
   if (Xmi_Select)
      {XrdCmsReq Req(this, Arg.Request.streamid);
       if (Xmi_Select->Select(&Req, XMI_LOCATE, Arg.Path, Arg.Opaque)) return 0;
      }

// Grab the refresh option (the only one we support)
//
   if (Arg.Opts & CmsLocateRequest::kYR_refresh) 
      {Sel.Opts  = XrdCmsSelect::Refresh; *toP++='s';}
   if (Arg.Opts & CmsLocateRequest::kYR_asap)
      {Sel.Opts |= XrdCmsSelect::Asap;    *toP++='i'; Sel.InfoP = &reqInfo;}
      else                                            Sel.InfoP = 0;

// Do some debugging
//
   *toP = '\0';
   DEBUGR(theopts <<' ' <<Arg.Path);

// Perform location
//
   if ((rc = Cluster.Locate(Sel)))
      {if (rc > 0)
          {Arg.Request.rrCode = kYR_wait;
           bytes = sizeof(Resp.Val); Why = "delay ";
          } else {
           if (rc == -2) return 0;
           Arg.Request.rrCode = kYR_error;
           rc = kYR_ENOENT; Why = "miss ";
           bytes = strlcpy(Resp.outbuff, "No servers have access to the file",
                   sizeof(Resp.outbuff)) + sizeof(Resp.Val) + 1;
          }
      } else {Why = "?"; bytes = 0;}

// List the servers
//
   if (!rc)
      {if (!Sel.Vec.hf || !(sP=Cluster.List(Sel.Vec.hf,XrdCmsCluster::LS_IPV6)))
          {Arg.Request.rrCode = kYR_error;
           rc = kYR_ENOENT; Why = "none ";
           bytes = strlcpy(Resp.outbuff, "No servers have the file",
                          sizeof(Resp.outbuff)) + sizeof(Resp.Val) + 1;
          } else rc = 0;
      }

// Either prepare to send an error or format the result
//
   if (rc)
      {Resp.Val           = htonl(rc);
       DEBUGR(Why <<Arg.Path);
      } else {
       bytes=do_LocFmt(Resp.outbuff,sP,Sel.Vec.pf,Sel.Vec.wf)+sizeof(Resp.Val)+1;
       Resp.Val            = 0;
       Arg.Request.rrCode  = kYR_data;
      }

// Send off the response
//
   Arg.Request.datalen = htons(bytes);
   ioV[1].iov_len      = bytes;
   Link->Send(ioV, 2, bytes+sizeof(Arg.Request));
   return 0;
}

/******************************************************************************/
/* Static                      d o _ L o c F m t                              */
/******************************************************************************/
  
int XrdCmsNode::do_LocFmt(char *buff, XrdCmsSelected *sP,
                          SMask_t pfVec, SMask_t wfVec)
{
   static const int Skip = (XrdCmsSelected::Disable | XrdCmsSelected::Offline);
   XrdCmsSelected *pP;
   char *oP = buff;

// format out the request as follows:                   
// 01234567810123456789212345678
// xy[::123.123.123.123]:123456
//
   while(sP)
        {if (!(sP->Status & Skip))
            {*oP     = (sP->Status & XrdCmsSelected::isMangr ? 'M' : 'S');
             if (sP->Mask & pfVec) *oP = tolower(*oP);
             *(oP+1) = (sP->Mask   & wfVec                   ? 'w' : 'r');
             strcpy(oP+2, sP->IPV6); oP += sP->IPV6Len + 2;
             if (sP->next) *oP++ = ' ';
            }
         pP = sP; sP = sP->next; delete pP;
        }

// Send of the result
//
   *oP = '\0';
   return (oP - buff);
}

/******************************************************************************/
/*                              d o _ M k d i r                               */
/******************************************************************************/
  
// Mkdir requests are forwarded to all subscribers subject to an Xmi callout.
//
const char *XrdCmsNode::do_Mkdir(XrdCmsRRData &Arg)
{
   EPNAME("do_Mkdir")
   mode_t mode = 0;
   int rc;

// Do some debugging
//
   DEBUGR("mode " <<Arg.Mode <<' ' <<Arg.Path);

// If we have an Xmi then call it
//
   if (Xmi_Mkdir)
      {XrdCmsReq Req(this, Arg.Request.streamid);
       if (!getMode(Arg.Mode, mode)) return "invalid mode";
          else if (Xmi_Mkdir->Mkdir(&Req, mode, Arg.Path, Arg.Opaque)) return 0;
      }

// We are don here if we have no data; otherwise convert the mode if we
// haven't done so already.
//
   if (!Config.DiskOK) return 0;
   if (!mode && !getMode(Arg.Mode, mode)) return "invalid mode";

// Attempt to create the directory either via call-out of oss plug-in
//
   if (Config.ProgMD) rc = fsExec(Config.ProgMD, Arg.Mode, Arg.Path);
      else rc = Config.ossFS->Mkdir(Arg.Path, mode);

// Return appropriate result
//
   return (rc ? fsFail(Arg.Ident, "mkdir", Arg.Path, rc) : 0);
}

/******************************************************************************/
/*                             d o _ M k p a t h                              */
/******************************************************************************/
  
// Mkpath requests are forwarded to all subscribers subjectto an Xmi callout.
//
const char *XrdCmsNode::do_Mkpath(XrdCmsRRData &Arg)
{
   EPNAME("do_Mkpath")
   mode_t mode = 0;
   int rc;

// Do some debugging
//
   DEBUGR("mode " <<Arg.Mode <<' ' <<Arg.Path);

// If we have an Xmi then call it
//
   if (Xmi_Mkpath)
      {XrdCmsReq Req(this, Arg.Request.streamid);
       if (!getMode(Arg.Mode, mode)) return "invalid mode";
          else if (Xmi_Mkpath->Mkpath(&Req,mode,Arg.Path,Arg.Opaque)) return 0;
      }

// We are don here if we have no data; otherwise convert the mode if we
// haven't done so already.
//
   if (!Config.DiskOK) return 0;
   if (!mode && !getMode(Arg.Mode, mode)) return "invalid mode";

// Attempt to create the directory path via call-out or oss plugin
//
   if (Config.ProgMP) rc = fsExec(Config.ProgMP, Arg.Mode, Arg.Path);
      else rc = Config.ossFS->Mkdir(Arg.Path, mode, 1);

// Return appropriate result
//
   return (rc ? fsFail(Arg.Ident, "mkpath", Arg.Path, rc) : 0);
}

/******************************************************************************/
/*                                 d o _ M v                                  */
/******************************************************************************/
  
// Mv requests are forwarded to all subscribers subject to an Xmi callout.
//
const char *XrdCmsNode::do_Mv(XrdCmsRRData &Arg)
{
   EPNAME("do_Mv")
   static const SMask_t allNodes(~0);
   int rc;

// Do some debugging
//
   DEBUGR(Arg.Path <<" to " <<Arg.Path2);

// If we have an Xmi then call it
//
   if (Xmi_Rename)
      {XrdCmsReq Req(this, Arg.Request.streamid, Arg.Request.modifier & kYR_dnf);
       if (Xmi_Rename->Rename(&Req,Arg.Path,Arg.Opaque,Arg.Path2,Arg.Opaque2))
          return 0;
      }

// If we are not a server, if must remove references to the old and new names
// from our cache. This is independent of how the raname is handled. We need
// not back percolate the mv since it was hanled top down in the first place.
// Note that we will scuttle the mv if the target file exists somewhere.
//
   if (!Config.DiskOK)
      {XrdCmsSelect Sel1(XrdCmsSelect::Defer, Arg.Path, strlen(Arg.Path ));
       XrdCmsSelect Sel2(XrdCmsSelect::Defer, Arg.Path2,strlen(Arg.Path2));

       // Setup select data (note that mv does not allow fast redirect)
       //
       Sel2.iovP = 0; Sel2.iovN = 0;
       Sel2.InfoP = 0;  // No fast redirects
       Sel2.nmask = SMask_t(0);

       // Perform selection
       //
       if ((rc = Cluster.Select(Sel2)))
          {if (rc > 0) {Arg.waitVal = rc; return "!mv";}
              else if (Sel2.Vec.hf)
                     {Say.Emsg("do_Mv",Arg.Path2,"exists; mv failed for",Arg.Path);
                      return "target file exists";
                     }
          }
       Cache.DelFile(Sel2, allNodes);
       Cache.DelFile(Sel1, allNodes);
       return 0;
      }
  
// Rename the file via call-out or oss plug-in (we used to do this via a requeue
// to the local xrootd but it's no longer necessary).
//
   if (Config.ProgMV) rc = fsExec(Config.ProgMV, Arg.Path, Arg.Path2);
      else rc = Config.ossFS->Rename(Arg.Path, Arg.Path2);

// Return appropriate result
//
   return (rc ? fsFail(Arg.Ident, "mv", Arg.Path, rc) : 0);
}

/******************************************************************************/
/*                               d o _ P i n g                                */
/******************************************************************************/
  
// Ping requests from a manager are local to the cell and never propagated.
//
const char *XrdCmsNode::do_Ping(XrdCmsRRData &Arg)
{
  static CmsPongRequest pongIt = {{0, kYR_pong, 0, 0}};

// Process: ping
// Respond: pong
//
   Link->Send((char *)&pongIt, sizeof(pongIt));
   return 0;
}
  
/******************************************************************************/
/*                               d o _ P o n g                                */
/******************************************************************************/
  
// Responses to a ping are local to the cell and never propagated.
//
const char *XrdCmsNode::do_Pong(XrdCmsRRData &Arg)
{
// Process: pong
// Reponds: n/a

   return 0;
}
  
/******************************************************************************/
/*                            d o _ P r e p A d d                             */
/******************************************************************************/
  
const char *XrdCmsNode::do_PrepAdd(XrdCmsRRData &Arg)
{
   EPNAME("do_PrepAdd")

// Do some debugging
//
   DEBUGR("parms: " <<Arg.Reqid <<' ' <<Arg.Notify <<' ' <<Arg.Prty <<' '
                    <<Arg.Mode  <<' ' <<Arg.Path);

// Do an Xmi callout if need be
//
   if (Xmi_Prep
   &&  Xmi_Prep->Prep(Arg.Reqid,
                      Arg.Opts & CmsPrepAddRequest::kYR_write ? XMI_RW : 0,
                      Arg.Path, Arg.Opaque))
       return 0;

// Queue this request for async processing
//
   (new XrdCmsPrepArgs(Arg))->Queue();
   return 0;
}
  
/******************************************************************************/
/*                            d o _ P r e p D e l                             */
/******************************************************************************/
  
const char *XrdCmsNode::do_PrepDel(XrdCmsRRData &Arg)
{
   EPNAME("do_PrepDel")

// Do some debugging
//
   DEBUGR("reqid " <<Arg.Reqid);

// Do a callout to the external manager if we have one
//
   if (Xmi_Prep && Xmi_Prep->Prep(Arg.Reqid, XMI_CANCEL, "", 0)) return 0;

// Cancel the request if applicable.
//
   if (Config.DiskOK)
      {if (!Config.DiskSS) {DEBUGR("ignoring cancel prepare " <<Arg.Reqid);}
          else {DEBUGR("canceling prepare " <<Arg.Reqid);
                PrepQ.Del(Arg.Reqid);
               }
      }
  return 0;
}
  
/******************************************************************************/
/*                                 d o _ R m                                  */
/******************************************************************************/
  
// Rm requests are forwarded to all subscribers subject to an Xmi callout.
//
const char *XrdCmsNode::do_Rm(XrdCmsRRData &Arg)
{
   EPNAME("do_Rm")
   static const SMask_t allNodes(~0);
   int rc;

// Do some debugging
//
   DEBUGR(Arg.Path);

// If we have an Xmi then call it
//
   if (Xmi_Remove)
      {XrdCmsReq Req(this, Arg.Request.streamid, Arg.Request.modifier & kYR_dnf);
       if (Xmi_Remove->Remove(&Req, Arg.Path, Arg.Opaque)) return 0;
      }

// If we have no data then we should remove this file from our cache
//
   if (!Config.DiskOK)
      {XrdCmsSelect Sel(0, Arg.Path, strlen(Arg.Path));
       Cache.DelFile(Sel, allNodes);
       return 0;
      }
  
// Remove the file either via call-out or the oss plugin. We used to requeue
// the request to the local xrootd but this is no longer needed.
//
   if (Config.ProgRM) rc = fsExec(Config.ProgRM, Arg.Path);
      else rc = Config.ossFS->Unlink(Arg.Path);

// Return appropriate result
//
   return (rc ? fsFail(Arg.Ident, "rm", Arg.Path, rc) : 0);
}
  
/******************************************************************************/
/*                              d o _ R m d i r                               */
/******************************************************************************/
  
// Rmdir requests are forwarded to all subscribers subject to an Xmi callout.
//
const char *XrdCmsNode::do_Rmdir(XrdCmsRRData &Arg)
{
   EPNAME("do_Rmdir")
   static const SMask_t allNodes(~0);
   int rc;

// Do some debugging
//
   DEBUGR(Arg.Path);

// If we have an Xmi then call it
//
   if (Xmi_Remdir)
      {XrdCmsReq Req(this, Arg.Request.streamid, Arg.Request.modifier & kYR_dnf);
       if (Xmi_Remdir->Remdir(&Req, Arg.Path, Arg.Opaque)) return 0;
      }

// If we have no data then we should remove this directory from our cache
//
   if (!Config.DiskOK)
      {XrdCmsSelect Sel(0, Arg.Path, strlen(Arg.Path));
       Cache.DelFile(Sel, allNodes);
       return 0;
      }
  
// Remove the directory either via call-out or the oss plug-in (we used to
// do this by requeing the request to the local xrootd; no longer needed).
//
   if (Config.ProgRD) rc = fsExec(Config.ProgRD, Arg.Path);
      else rc = Config.ossFS->Remdir(Arg.Path);

// Return appropriate result
//
   return (rc ? fsFail(Arg.Ident, "rmdir", Arg.Path, rc) : 0);
}
  
/******************************************************************************/
/*                             d o _ S e l e c t                              */
/******************************************************************************/
  
// A select request comes from a redirector and is handled locally within the
// cell. This may cause "state" requests to be broadcast to subscribers.
//
const char *XrdCmsNode::do_Select(XrdCmsRRData &Arg)
{
   EPNAME("do_Select")
   XrdCmsRRQInfo reqInfo(Instance, RSlot, Arg.Request.streamid);
   XrdCmsSelect Sel(XrdCmsSelect::Peers, Arg.Path, Arg.PathLen-1);
   struct iovec ioV[2];
   char theopts[16], *Avoid, *toP = theopts;
   int rc, bytes;

// Do a callout to the external manager if we have one
//
   if (Arg.Opts & CmsSelectRequest::kYR_stat && Xmi_Stat)
      {XrdCmsReq Req(this, Arg.Request.streamid);
       if (Xmi_Stat->Stat(&Req, Arg.Path, Arg.Opaque)) return 0;
      } else 
   if (Xmi_Select)
      {XrdCmsReq Req(this, Arg.Request.streamid);
       int opts = (Arg.Opts & CmsSelectRequest::kYR_write ? XMI_RW : 0);
       if (Arg.Opts & CmsSelectRequest::kYR_create) opts |= XMI_NEW;
       if (Arg.Opts & CmsSelectRequest::kYR_trunc)  opts |= XMI_TRUNC;
       if (Xmi_Select->Select(&Req, opts, Arg.Path, Arg.Opaque)) return 0;
      }

// Init select data (note that refresh supresses fast redirects)
//
   Sel.iovP  = 0; Sel.iovN  = 0; Sel.InfoP = &reqInfo;

// Complete the arguments to select
//
         if (Arg.Opts & CmsSelectRequest::kYR_refresh) 
           {Sel.Opts |= XrdCmsSelect::Refresh;                     *toP++='s';}
         if (Arg.Opts & CmsSelectRequest::kYR_online)  
           {Sel.Opts |= XrdCmsSelect::Online;                      *toP++='o';}
         if (Arg.Opts & CmsSelectRequest::kYR_stat)
           {Sel.Opts |= XrdCmsSelect::noBind;                      *toP++='x';}
   else {if (Arg.Opts & CmsSelectRequest::kYR_trunc)   
           {Sel.Opts |= XrdCmsSelect::Write | XrdCmsSelect::Trunc; *toP++='t';}
         if (Arg.Opts & CmsSelectRequest::kYR_write)   
           {Sel.Opts |= XrdCmsSelect::Write;                       *toP++='w';}
         if (Arg.Opts & CmsSelectRequest::kYR_metaop)
           {Sel.Opts |= XrdCmsSelect::Write|XrdCmsSelect::isMeta;  *toP++='m';}
         if (Arg.Opts & CmsSelectRequest::kYR_create)  
           {Sel.Opts |= XrdCmsSelect::Write|XrdCmsSelect::NewFile; *toP++='c';
            if (Arg.Opts & CmsSelectRequest::kYR_replica)
               {Sel.Opts |= XrdCmsSelect::Replica;                 *toP++='+';}
           }
        }
   *toP = '\0';

// Check if an avoid node present. If so, this is ineligible for fast redirect.
//
   Sel.nmask = SMask_t(0);
   if ((Avoid = Arg.Avoid))
      {unsigned int IPaddr;
       char *Comma;
       DEBUGR(theopts <<' ' <<Arg.Path <<" avoiding " <<Avoid);
       Sel.InfoP = 0;
       do {if ((Comma = index(Avoid,','))) *Comma = '\0';
           if (*Avoid == '+') Sel.nmask |= Cluster.getMask(Avoid+1);
              else if (XrdNetDNS::Host2IP(Avoid, &IPaddr))
                              Sel.nmask |= Cluster.getMask(IPaddr);
           Avoid = Comma+1;
          } while(Comma && *Avoid);
      } else DEBUGR(theopts <<' ' <<Arg.Path);

// Perform selection
//
   if ((rc = Cluster.Select(Sel)))
      {if (rc > 0)
          {Arg.Request.rrCode = kYR_wait;
           Sel.Resp.Port      = rc;
           Sel.Resp.DLen      = 0;
           DEBUGR("delay " <<rc <<' ' <<Arg.Path);
          } else {
           Arg.Request.rrCode = kYR_error;
           Sel.Resp.Port      = kYR_ENOENT;
           DEBUGR("failed; " <<Sel.Resp.Data << ' ' <<Arg.Path);
          }
      } else if (!Sel.Resp.DLen) return 0;
                else {Arg.Request.rrCode = kYR_redirect;
                      DEBUGR("Redirect -> " <<Sel.Resp.Data <<':'
                            <<Sel.Resp.Port <<" for " <<Arg.Path);
             }

// Format the response
//
   bytes               = Sel.Resp.DLen+sizeof(Sel.Resp.Port);
   Arg.Request.datalen = htons(bytes);
   Sel.Resp.Port       = htonl(Sel.Resp.Port);

// Fill out the I/O vector
//
   ioV[0].iov_base = (char *)&Arg.Request;
   ioV[0].iov_len  = sizeof(Arg.Request);
   ioV[1].iov_base = (char *)&Sel.Resp;
   ioV[1].iov_len  = bytes;

// Send back the response
//
   Link->Send(ioV, 2, bytes+sizeof(Arg.Request));
   return 0;
}
  
/******************************************************************************/
/*                            d o _ S e l P r e p                             */
/******************************************************************************/
  
int XrdCmsNode::do_SelPrep(XrdCmsPrepArgs &Arg) // Static!!!
{
   EPNAME("do_SelPrep")
   XrdCmsSelect Sel(XrdCmsSelect::Peers, Arg.path, Arg.pathlen-1);
   int rc;

// Do a callout to the external manager if we have one
//
   if (Xmi_Prep)
      {int opts = (Arg.options & CmsPrepAddRequest::kYR_write ? XMI_RW : 0);
       if (Xmi_Prep->Prep(Arg.reqid, opts, Arg.path, Arg.opaque)) return 0;
      }

// Complete the arguments to select
//
   if ( Arg.options & CmsPrepAddRequest::kYR_fresh)
      Sel.Opts |= XrdCmsSelect::Freshen;
   if ( Arg.options & CmsPrepAddRequest::kYR_write) 
      Sel.Opts |= XrdCmsSelect::Write;
   if (Arg.options & CmsPrepAddRequest::kYR_stage) 
           {Sel.iovP = Arg.ioV; Sel.iovN = Arg.iovNum;}
      else {Sel.iovP = 0;       Sel.iovN = 0;
            Sel.Opts |= XrdCmsSelect::Defer;
           }

// Setup select data (note that prepare does not allow fast redirect)
//
   Sel.InfoP = 0;  // No fast redirects
   Sel.nmask = SMask_t(0);

// Check if co-location wanted relevant only when staging wanted
//
   if (Arg.clPath && Sel.iovP)
      {XrdCmsSelect Scl(XrdCmsSelect::Peers, Arg.clPath, strlen(Arg.clPath));
       Scl.iovP = 0; Scl.iovN  = 0; Scl.InfoP = 0; Scl.nmask = SMask_t(0);
       DEBUGR("colocating " <<Arg.path <<" w.r.t. " <<Arg.clPath);
       rc = Cluster.Select(Scl);
       if (rc > 0) {Sched->Schedule((XrdJob *)&Arg, rc+time(0));
                    DEBUGR("coloc to " <<Arg.clPath <<" delayed " <<rc <<" seconds");
                    return 1;
                   }
       if (rc < 0) Say.Emsg("SelPrep", Arg.path, "failed;", Sel.Resp.Data);
          else Sel.nmask = ~Scl.smask;
      }

// Perform selection
//
   if ((rc = Cluster.Select(Sel)))
      {if (rc > 0)
          {if (!(Arg.options & CmsPrepAddRequest::kYR_stage)) return 0;
           Sched->Schedule((XrdJob *)&Arg, rc+time(0));
           DEBUGR("prep delayed " <<rc <<" seconds");
           return 1;
          }
       Say.Emsg("SelPrep", Arg.path, "failed;", Sel.Resp.Data);
       PrepQ.Inform("unavail", &Arg);
      }

// All done
//
   return 0;
}

/******************************************************************************/
/*                              d o _ S p a c e                               */
/******************************************************************************/
  
// Manager space requests are local to the cell and never propagated.
//
const char *XrdCmsNode::do_Space(XrdCmsRRData &Arg)
{
   EPNAME("do_Space")
   struct iovec xmsg[2];
   CmsAvailRequest mySpace = {{0, kYR_avail, 0, 0}};
   char         buff[sizeof(int)*2+2], *bp = buff;
   int blen, maxfr, tutil;

// Process: <id> space
// Respond: <id> avail <numkb> <dskutil>
//
   maxfr = Meter.FreeSpace(tutil);

// Do some debugging
//
   DEBUGR(maxfr <<"MB free; " <<tutil <<"% util");

// Construct a message to be sent to the manager.
//
   blen  = XrdOucPup::Pack(&bp, maxfr);
   blen += XrdOucPup::Pack(&bp, tutil);
   mySpace.Hdr.datalen = htons(static_cast<unsigned short>(blen));

// Send the response
//
   if (Arg.Request.rrCode != kYR_space) Manager.Inform(mySpace.Hdr, buff, blen);
      else {xmsg[0].iov_base = (char *)&mySpace;
            xmsg[0].iov_len  = sizeof(mySpace);
            xmsg[1].iov_base = buff;
            xmsg[1].iov_len  = blen;
            mySpace.Hdr.datalen = htons(static_cast<unsigned short>(blen));
            Link->Send(xmsg, 2);
           }
   return 0;
}
  
/******************************************************************************/
/*                              d o _ S t a t e                               */
/******************************************************************************/
  
// State requests from a manager are rebroadcast to all relevant subscribers.
//
const char *XrdCmsNode::do_State(XrdCmsRRData &Arg)
{
   EPNAME("do_State")
   struct iovec xmsg[2];
   int noResp = Arg.Request.modifier & CmsStateRequest::kYR_noresp;

// Do some debugging
//
   TRACER(Files,Arg.Path);

// Process: state <path>
// Respond: have <path>
//
   isKnown = 1;

// If we are a manager then check for the file in the local cache. Otherwise,
// if we have actual data, do a stat() on the file.
//
   if (isMan) Arg.Request.modifier = do_StateFWD(Arg);
      else if (Config.DiskOK)
              Arg.Request.modifier = isOnline(Arg.Path, 0);
              else return 0;

// Respond appropriately
//
   if (Arg.Request.modifier && !noResp)
      {xmsg[0].iov_base      = (char *)&Arg.Request;
       xmsg[0].iov_len       = sizeof(Arg.Request);
       xmsg[1].iov_base      = Arg.Buff;
       xmsg[1].iov_len       = Arg.Dlen;
       Arg.Request.rrCode    = kYR_have;
       Arg.Request.modifier |= kYR_raw;
       Link->Send(xmsg, 2);
      }
   return 0;
}
  
/******************************************************************************/
/*                           d o _ S t a t e F W D                            */
/******************************************************************************/
  
int XrdCmsNode::do_StateFWD(XrdCmsRRData &Arg)
{
   EPNAME("do_StateFWD");
   XrdCmsSelect Sel(0, Arg.Path, Arg.PathLen-1);
   XrdCmsPInfo  pinfo;
   int retc;

// Find out who could serve this file
//
   if (!Cache.Paths.Find(Arg.Path, pinfo) || pinfo.rovec == 0)
      {DEBUGR("Path find failed for state " <<Arg.Path);
       return 0;
      }

// Get the primary locations for this file
//
   Sel.Vec.hf = Sel.Vec.pf = Sel.Vec.bf = 0;
   if (Arg.Request.modifier & CmsStateRequest::kYR_refresh) retc = 0;
      else retc = Cache.GetFile(Sel, pinfo.rovec);

// If need be, ask the relevant nodes if they have the file.
//
   if (!retc || Sel.Vec.bf != 0)
      {if (!retc) Cache.AddFile(Sel, 0);
       Cluster.Broadcast((retc ? Sel.Vec.bf : pinfo.rovec), Arg.Request,
                         (void *)Arg.Buff, Arg.Dlen);
      }

// Return true if anyone has the file at this point
//
   if (Sel.Vec.hf != 0) return CmsHaveRequest::Online;
   if (Sel.Vec.pf != 0) return CmsHaveRequest::Pending;
                        return 0;
}
  
/******************************************************************************/
/*                             d o _ S t a t F S                              */
/******************************************************************************/
  
const char *XrdCmsNode::do_StatFS(XrdCmsRRData &Arg)
{
   static kXR_unt32 Zero = 0;
   char         buff[256];
   struct iovec ioV[3] = {{(char *)&Arg.Request, sizeof(Arg.Request)},
                          {(char *)&Zero,        sizeof(Zero)},
                          {(char *)&buff,        0}};
   XrdCmsPInfo   pinfo;
   int           bytes;
   SpaceData     theSpace;

// Find out who serves this path and get space relative to it
//
   if (Cache.Paths.Find(Arg.Path, pinfo) && pinfo.rovec)
      {Cluster.Space(theSpace, pinfo.rovec);
       bytes = sprintf(buff, "%d %d %d %d %d %d",
                       theSpace.wNum, theSpace.wFree>>10, theSpace.wUtil,
                       theSpace.sNum, theSpace.sFree>>10, theSpace.sUtil) + 1;
      } else bytes = strlcpy(buff, "-1 -1 -1 -1 -1 -1", sizeof(buff)) + 1;

// Send the response
//
   ioV[2].iov_len      = bytes;
   bytes              += sizeof(Zero);
   Arg.Request.rrCode  = kYR_data;
   Arg.Request.datalen = htons(bytes);
   Link->Send(ioV, 3, bytes+sizeof(Arg.Request));
   return 0;
}

/******************************************************************************/
/*                              d o _ S t a t s                               */
/******************************************************************************/
  
// We punt on stats requests as we have no way to export them anyway.
//
const char *XrdCmsNode::do_Stats(XrdCmsRRData &Arg)
{
   static const unsigned short szLen = sizeof(kXR_unt32);
   static XrdSysMutex StatsData;
   static int         statsz = 0;
   static int         statln = 0;
   static char       *statbuff = 0;
   static time_t      statlast = 0;
   static kXR_unt32   theSize;

   struct iovec  ioV[3] = {{(char *)&Arg.Request, sizeof(Arg.Request)},
                           {(char *)&theSize,     sizeof(theSize)},
                           {0,                    0}
                          };
   time_t tNow;

// Allocate buffer if we do not have one
//
   StatsData.Lock();
   if (!statsz || !statbuff)
      {statsz   = Cluster.Stats(0,0);
       statbuff = (char *)malloc(statsz);
       theSize = htonl(statsz);
      }

// Check if only the size is wanted
//
   if (Arg.Request.modifier & CmsStatsRequest::kYR_size)
      {ioV[1].iov_len = sizeof(theSize);
       Arg.Request.datalen = htons(szLen);
       Arg.Request.rrCode  = kYR_data;
       Link->Send(ioV, 2);
       StatsData.UnLock();
       return 0;
      }

// Get full statistics if enough time has passed
//
   tNow = time(0);
   if (statlast+9 >= tNow)
      {statln = Cluster.Stats(statbuff, statsz); statlast = tNow;}

// Format result and send response
//
   ioV[2].iov_base = statbuff;
   ioV[2].iov_len  = statln;
   Arg.Request.datalen = htons(static_cast<unsigned short>(szLen+statln));
   Arg.Request.rrCode  = kYR_data;
   Link->Send(ioV, 3);

// All done
//
   StatsData.UnLock();
   return 0;
}

/******************************************************************************/
/*                             d o _ S t a t u s                              */
/******************************************************************************/
  
// the reset request is propagated to all of our managers. A special reset case
// is sent when a subscribed supervisor adds a new node. This causes all cache
// lines for the supervisor to be marked suspect. Status change requests are
// propagated to upper-level managers only if the summary state has changed.
//
const char *XrdCmsNode::do_Status(XrdCmsRRData &Arg)
{
   EPNAME("do_Status")
   const char *srvMsg, *stgMsg;
   int   Stage = Arg.Request.modifier & CmsStatusRequest::kYR_Stage;
   int noStage = Arg.Request.modifier & CmsStatusRequest::kYR_noStage;
   int Resume  = Arg.Request.modifier & CmsStatusRequest::kYR_Resume;
   int Suspend = Arg.Request.modifier & CmsStatusRequest::kYR_Suspend;
   int Reset   = Arg.Request.modifier & CmsStatusRequest::kYR_Reset;
   int add2Activ, add2Stage;

// Do some debugging
//
   DEBUGR(  (Reset  ? "reset " : "")
          <<(Resume ? "resume " : (Suspend ? "suspend " : ""))
          <<(Stage  ? "stage "  : (noStage ? "nostage " : "")));

// Process reset requests. These are exclsuive to any other request
//
   if (Reset)
      {Manager.Reset();                // Propagate the reset to our managers
       Cache.Bounce(NodeMask, NodeID); // Now invalidate our cache lines
      }

// Process stage/nostage
//
    if ((Stage && isNoStage) || (noStage && !isNoStage))
       if (noStage) {add2Stage = -1; isNoStage = 1; stgMsg="staging suspended";}
          else      {add2Stage =  1; isNoStage = 0; stgMsg="staging resumed";}
       else         {add2Stage =  0;                stgMsg = 0;}

// Process suspend/resume
//
    if ((Resume && isSuspend) || (Suspend && !isSuspend))
       if (Suspend) {add2Activ = -1; isSuspend = 1;
                     srvMsg="service suspended"; 
                     stgMsg = 0;
                    }
          else      {add2Activ =  1; isSuspend = 0;
                     srvMsg="service resumed";
                     stgMsg = (isNoStage ? "(no staging)" : "(staging)");
                     Port = ntohl(Arg.Request.streamid);
                     DEBUGR("set data port to " <<Port);
                    }
       else         {add2Activ =  0; srvMsg = 0;}

// Get the most important message out
//
   if (isOffline)         {srvMsg = "service offline";  stgMsg = 0;}
      else if (isDisable) {srvMsg = "service disabled"; stgMsg = 0;}

// Now see if we need to change anything
//
   if (add2Activ || add2Stage)
       {CmsState.Update(XrdCmsState::Counts, add2Activ, add2Stage);
        Say.Emsg("Node", Name(), srvMsg, stgMsg);
       }

   return 0;
}

/******************************************************************************/
/*                              d o _ T r u n c                               */
/******************************************************************************/
  
// Trunc requests are forwarded to all subscribers subject to an Xmi callout.
// Currently, we have no definition in Xmi for trunc!
//
const char *XrdCmsNode::do_Trunc(XrdCmsRRData &Arg)
{
   EPNAME("do_Trunc")
   long long Size = -1;
   int rc;

// Do some debugging
//
   DEBUGR("size " <<Arg.Mode <<' ' <<Arg.Path);

// If we have an Xmi then call it
//
// if (Xmi_Trunc)
//    {XrdCmsReq Req(this, Arg.Request.streamid);
//     if (!getSize(Arg.Mode, Size)) return "invalid size";
//        else if (Xmi_Trunc->Trunc(&Req, Size, Arg.Path, Arg.Opaque)) return 0;
//    }

// We are don here if we have no data; otherwise convert the mode if we
// haven't done so already.
//
   if (!Config.DiskOK) return 0;
   if (Size < 0 && !getSize(Arg.Mode, Size)) return "invalid size";

// Attempt to change the size either via call-out or the oss plug-in
//
   if (Config.ProgTR) rc = fsExec(Config.ProgTR, Arg.Mode, Arg.Path);
      else rc = Config.ossFS->Truncate(Arg.Path, Size);

// Return appropriate result
//
   return (rc ? fsFail(Arg.Ident, "trunc", Arg.Path, rc) : 0);
}

/******************************************************************************/
/*                                d o _ T r y                                 */
/******************************************************************************/

// Try requests from a manager indicate that we are being displaced and should
// hunt for another manager. The request provides hints as to where to try.
//
const char *XrdCmsNode::do_Try(XrdCmsRRData &Arg)
{
   EPNAME("do_Try")
   XrdOucTokenizer theList(Arg.Path);
   char *tp;

// Do somde debugging
//
   DEBUGR(Arg.Path);

// Delete any additions from this manager
//
   myMans.Del(IPAddr);

// Add all the alternates to our alternate list
//
   tp = theList.GetLine();
   while((tp = theList.GetToken()))
         myMans.Add(IPAddr, tp, Config.PortTCP, myLevel);

// Close the link and return an error
//
// Disc("redirected.");
   return ".redirected";
}
  
/******************************************************************************/
/*                             d o _ U p d a t e                              */
/******************************************************************************/
  
const char *XrdCmsNode::do_Update(XrdCmsRRData &Arg)
{

// Process: <id> update
// Respond: <id> status
//
   CmsState.sendState(Link);
   return 0;
}
  
/******************************************************************************/
/*                              d o _ U s a g e                               */
/******************************************************************************/
  
// Usage requests from a manager are local to the cell and never propagated.
//
const char *XrdCmsNode::do_Usage(XrdCmsRRData &Arg)
{

// Process: <id> usage
// Respond: <id> load <cpu> <io> <load> <mem> <pag> <dskfree> <dskutil>
//
   Report_Usage(Link);
   return 0;
}

/******************************************************************************/
/*                          R e p o r t _ U s a g e                           */
/******************************************************************************/
  
void XrdCmsNode::Report_Usage(XrdLink *lp)   // Static!
{
   EPNAME("Report_Usage")
   CmsLoadRequest myLoad = {{0, kYR_load, 0, 0}};
   struct iovec xmsg[2];
   char loadbuff[CmsLoadRequest::numLoad];
   char respbuff[sizeof(loadbuff)+2+sizeof(int)+2], *bp = respbuff;
   int  blen, maxfr, pcpu, pnet, pxeq, pmem, ppag, pdsk;

// Respond: <id> load <cpu> <io> <load> <mem> <pag> <dskfree> <dskutil>
//
   maxfr = Meter.Report(pcpu, pnet, pxeq, pmem, ppag, pdsk);

   loadbuff[CmsLoadRequest::cpuLoad] = static_cast<char>(pcpu);
   loadbuff[CmsLoadRequest::netLoad] = static_cast<char>(pnet);
   loadbuff[CmsLoadRequest::xeqLoad] = static_cast<char>(pxeq);
   loadbuff[CmsLoadRequest::memLoad] = static_cast<char>(pmem);
   loadbuff[CmsLoadRequest::pagLoad] = static_cast<char>(ppag);
   loadbuff[CmsLoadRequest::dskLoad] = static_cast<char>(pdsk);

   blen  = XrdOucPup::Pack(&bp, loadbuff, sizeof(loadbuff));
   blen += XrdOucPup::Pack(&bp, maxfr);
   myLoad.Hdr.datalen = htons(static_cast<unsigned short>(blen));

   xmsg[0].iov_base = (char *)&myLoad;
   xmsg[0].iov_len  = sizeof(myLoad);
   xmsg[1].iov_base = respbuff;
   xmsg[1].iov_len  = blen;
   if (lp) lp->Send(xmsg, 2);
      else Manager.Inform("usage", xmsg, 2);

// Do some debugging
//
   DEBUG("cpu=" <<pcpu <<" net=" <<pnet <<" xeq=" <<pxeq
      <<" mem=" <<pmem <<" pag=" <<ppag <<" dsk=" <<pdsk <<' ' <<maxfr);
}
  
/******************************************************************************/
/*                             S y n c S p a c e                              */
/******************************************************************************/

void XrdCmsNode::SyncSpace()
{
   XrdCmsRRData Arg;
   int          old_free = 0;

// For newly logged in nodes, we need to sync the free space stats
//
   mlMutex.Lock();
   if (isRW && DiskFree > LastFree)
      {old_free = LastFree; LastFree = DiskFree;}
   mlMutex.UnLock();

// Tell our manager if we now have more space, if need be.
//
   if (!old_free)
      {Arg.Request.rrCode = kYR_login;
       Arg.Ident   = Ident;
       Arg.dskFree = DiskFree;
       Arg.dskUtil = DiskUtil;
       do_Space(Arg);
      }
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                f s E x e c                                 */
/******************************************************************************/

int XrdCmsNode::fsExec(XrdOucProg *Prog, char *Arg1, char *Arg2)
{
   static const int PfnSZ = XrdCmsMAX_PATH_LEN+1;
   char Pfn1[PfnSZ], Pfn2[PfnSZ];

// The first argument may or may not be a path. The second, if present is always
// a path. If we have a name mapper then we need to substitute remapped paths.
//
   if (Config.lcl_N2N)
      {if (*Arg1 == '/')
          {if (Config.lcl_N2N->lfn2pfn(Arg1,Pfn1,PfnSZ-1)) return fsL2PFail1;
           Arg1 = Pfn1;
          }
       if ( Arg2)
          {if (Config.lcl_N2N->lfn2pfn(Arg2,Pfn2,PfnSZ-1)) return fsL2PFail2;
           Arg2 = Pfn2;
          }
      }

// Return results of the call-out
//
   return Prog->Run(Arg1, Arg2);
}
  
/******************************************************************************/
/*                                f s F a i l                                 */
/******************************************************************************/
  
const char *XrdCmsNode::fsFail(const char *Who,  const char *What,
                               const char *Path, int rc)
{
   EPNAME("fsFail")

// Immediately return on the two most unlikely failures; o/w issue message
//
   rc = abs(rc);
   if (rc == fsL2PFail1) return "lfn2pfn path1 failed";
   if (rc == fsL2PFail2) return "lfn2pfn path2 failed";
   if (rc != ENOENT) Say.Emsg("Node", rc, What, Path);
      else {struct {const char *Ident;} Arg = {Who};
            DEBUGR("rc=" <<rc <<' ' <<What <<' ' <<Path);
           }
   return rc ? strerror(rc) : 0;
}

/******************************************************************************/
/*                               g e t M o d e                                */
/******************************************************************************/

int XrdCmsNode::getMode(const char *theMode, mode_t &Mode)
{
   char *eP;

// Convert the mode argument
//
   if (!(Mode = strtol(theMode, &eP, 8)) || *eP || (Mode >> 9)) return 0;
   return 1;
}

/******************************************************************************/
/*                               g e t S i z e                                */
/******************************************************************************/

int XrdCmsNode::getSize(const char *theSize, long long &Size)
{
   char *eP;

// Convert the size argument
//
   if (!(Size = strtoll(theSize, &eP, 10)) || *eP) return 0;
   return 1;
}
  
/******************************************************************************/
/*                              i s O n l i n e                               */
/******************************************************************************/
  
int XrdCmsNode::isOnline(char *path, int upt) // Static!!!
{
   static const int Sopts = XRDOSS_resonly | XRDOSS_updtatm;
   struct stat buf;

// Issue stat() via oss plugin. If it fails. the file may still exist in the
// prepare queue (i.e., logically present).
//
   if (Config.ossFS->Stat(path, &buf, Sopts))
      {if (Config.DiskSS && PrepQ.Exists(path)) return CmsHaveRequest::Pending;
          else return 0;
      }

// Determine what to return
// the update if an oss plugin exists as we don't have a way of doing this then.
//
   if ((buf.st_mode & S_IFMT) == S_IFREG)
       return (buf.st_mode & S_ISUID ? CmsHaveRequest::Pending
                                     : CmsHaveRequest::Online);

   return (buf.st_mode & S_IFMT) == S_IFDIR ? CmsHaveRequest::Online : 0;
}
