/******************************************************************************/
/*                                                                            */
/*                     X r d O l b P r o t o c o l . c c                      */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

const char *XrdOlbProtocolCVSID = "$Id$";
 
  
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <iostream.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>

#include "Xrd/XrdInet.hh"
#include "Xrd/XrdLink.hh"
#include "XrdNet/XrdNetLink.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdOlb/XrdOlbConfig.hh"
#include "XrdOlb/XrdOlbManager.hh"
#include "XrdOlb/XrdOlbProtocol.hh"
#include "XrdOlb/XrdOlbTrace.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"

using namespace XrdOlb;
  
/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

       XrdSysError      XrdOlb::Say(0, "olb_");

       XrdOucTrace      XrdOlb::Trace(&XrdOlb::Say);

       XrdSysMutex      XrdOlbProtocol::ProtMutex;
       XrdOlbProtocol  *XrdOlbProtocol::ProtStack = 0;

       XrdInet         *XrdOlbProtocol::myNet     = 0;
       int              XrdOlbProtocol::readWait  = 1000;

/******************************************************************************/
/*                       P r o t o c o l   L o a d e r                        */
/*                        X r d g e t P r o t o c o l                         */
/******************************************************************************/
  
// This protocol can live in a shared library. It can also be statically linked
// to provide a default protocol (which, for olb protocol we do). The interface
// below is used by the server to obtain a copy of the protocol object that can
// be used to decide whether or not a link is talking our particular protocol.
// Phase 1 initialization occured on the call to XrdgetProtocolPort(). At this
// point a network interface is defined and we can complete initialization.
//

extern "C"
{
XrdProtocol *XrdgetProtocol(const char *pname, char *parms,
                            XrdProtocol_Config *pi)
{

// If an error occured in Phase 1 then we must return a null object to force
// termination of this instance (with a meaningful message).
//
   if (Config.Disabled < 0) return (XrdProtocol *)0;

// Initialize the network interface and get the actual port number assigned
//
   XrdOlbProtocol::setNet(pi->NetTCP, pi->readWait);
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
   return (XrdProtocol *)new XrdOlbProtocol();
}
}

/******************************************************************************/
/*           P r o t o c o l   P o r t   D e t e r m i n a t i o n            */
/*                    X r d g e t P r o t o c o l P o r t                     */
/******************************************************************************/
  
// Because the olb port numbers are determined dynamically based on the role the
// olb plays, we need to process the configration file and return the right
// port number if it differs from the one provided by the protocol driver. Only
// one port instance of the olbd protocol is allowed.
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
           Say.Emsg("Config", "Alternate olbd at port", buff);
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
   Say.Say("Copr.  2006 Stanford University/SLAC olbd.");

// Return an arbitrary port if static init fails. We will return true failure
// when the protocol driver tries to get the first protocol object.
//
   if (cfn) cfn = strdup(cfn);
   if (Config.Configure1(pi->argc, pi->argv, cfn)) 
      {Config.Disabled = -1;  return 0;}

// Return the port number to be used
//
   thePort = Config.PortTCP;
   return thePort;
}
}

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/

XrdOlbProtocol *XrdOlbProtocol::Alloc()
{
XrdOlbProtocol *xp;

// Grab a protocol object and, if none, return a new one
//
   ProtMutex.Lock();
   if ((xp = ProtStack)) ProtStack = xp->ProtLink;
      else xp = new XrdOlbProtocol();
   ProtMutex.UnLock();

// All done
//
   return xp;
}
  
/******************************************************************************/
/*                                 M a t c h                                  */
/******************************************************************************/

XrdProtocol *XrdOlbProtocol::Match(XrdLink *lp)
{
char hsbuff[6];
int dlen;

// Peek at the first 6 bytes of data (shouldb be "login ")
//
   if ((dlen = lp->Peek(hsbuff,sizeof(hsbuff),readWait)) != sizeof(hsbuff))
      {if (dlen <= 0) lp->setEtext("login not received");
       return (XrdProtocol *)0;
      }

// Verify that this is our protocol
//
   if (strncmp(hsbuff, "login ", sizeof(hsbuff))) return 0;

// Return the protocol object
//
   return (XrdProtocol *)XrdOlbProtocol::Alloc();
}
 
/******************************************************************************/
/*                               P r o c e s s                                */
/******************************************************************************/
  
// Process is called only when we get a new connection. We only return when
// the connection drops. So, we just scatle off to the login process.
//
int XrdOlbProtocol::Process(XrdLink *lp)
{
   XrdNetPeer netPeer;
   XrdNetLink *np;

// First we must convert an XrdLink object to an XrdNetLink object for
// historic reasons to get the extra functionality. We do this by first
// constructing a XrdNetPeer object and using it to get an XrdNetLink object
// The NOCLOSE option indicates that the fd is shared between two objects.
//
   netPeer.fd = lp->FDnum();
   lp->Name(&netPeer.InetAddr);
   netPeer.InetName = (char *)lp->Host();
   np = XrdNetLink::Alloc(&Say,(XrdNet *)myNet,netPeer,0,XRDNETLINK_NOCLOSE);

// We must make sure that the InetName pointer is cleared in the netPeer
// object because it's not private data. Then check for allocation failure.
//
   netPeer.InetName = 0;
   if (!np) {lp->setEtext("NetLink allocation failure"); return -1;}

// Now process the login (link is recycled internally -- this is historical)
//
   Manager.Login(np);

// All done indicate the connection is dead
//
   return -1;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdOlbProtocol::Recycle(XrdLink *lp, int csec, const char *reason)
{

// Push ourselves on the stack
//
   ProtMutex.Lock();
   ProtLink  = ProtStack;
   ProtStack = this;
   ProtMutex.UnLock();
}
