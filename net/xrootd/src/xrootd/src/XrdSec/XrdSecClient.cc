/******************************************************************************/
/*                                                                            */
/*                       X r d S e c C l i e n t . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSec/XrdSecPManager.hh"
#include "XrdSec/XrdSecInterface.hh"

/******************************************************************************/
/*                 M i s c e l l a n e o u s   D e f i n e s                  */
/******************************************************************************/

#define DEBUG(x) {if (DebugON) cerr <<"sec_Client: " <<x <<endl;}

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
class XrdSecProtNone : public XrdSecProtocol
{
public:
int                Authenticate  (XrdSecCredentials  *cred,
                                  XrdSecParameters  **parms,
                                  XrdOucErrInfo      *einfo=0) 
                                 {return 0;}

XrdSecCredentials *getCredentials(XrdSecParameters  *parm=0,       // In
                                  XrdOucErrInfo     *einfo=0)
                                 {return new XrdSecCredentials();}

void               Delete() {}  // Never deleted because it's static!

              XrdSecProtNone() : XrdSecProtocol("") {}
             ~XrdSecProtNone() {}
};
  
/******************************************************************************/
/*                     X r d S e c G e t P r o t o c o l                      */
/******************************************************************************/

// This function is only invoked by the client. It exists in the top level
// shared library that interposes between all other protocol shared libraries.
//
extern "C"
{
XrdSecProtocol *XrdSecGetProtocol(const char             *hostname,
                                  const struct sockaddr  &netaddr,
                                        XrdSecParameters &parms,
                                        XrdOucErrInfo    *einfo)
{
   static int DebugON = ((getenv("XrdSecDEBUG") &&
                          strcmp(getenv("XrdSecDEBUG"), "0")) ? 1 : 0);
   static XrdSecProtNone ProtNone;
   static XrdSecPManager PManager(DebugON);
   const char *noperr = "XrdSec: No authentication protocols are available.";

   XrdSecProtocol *protp;

// Perform any required debugging
//
   DEBUG("protocol request for host " <<hostname <<" token='"
         <<(parms.size > 0 ? parms.buffer : "") <<"'");

// Check if the server wants no security.
//
   if (!parms.size || !parms.buffer[0]) return (XrdSecProtocol *)&ProtNone;

// Find a supported protocol.
//
   if (!(protp = PManager.Get(hostname, netaddr, parms)))
      {if (einfo) einfo->setErrInfo(ENOPROTOOPT, noperr);
         else cerr <<noperr <<endl;
      }

// All done
//
   return protp;
}
}
