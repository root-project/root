/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l h o s t . c c                  */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$
  
const char *XrdSecProtocolhostCVSID = "$Id$";

#include <strings.h>
#include <stdlib.h>

#include "XrdSec/XrdSecProtocolhost.hh"

/******************************************************************************/
/*                          A u t h e n t i c a t e                           */
/******************************************************************************/
  
int XrdSecProtocolhost::Authenticate(XrdSecCredentials  *cred,
                                     XrdSecParameters  **parms,
                                     XrdOucErrInfo      *einfo)
{
   strcpy(Entity.prot, "host");
   Entity.host = theHost;
   return 0;
}

/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/
  
XrdSecCredentials *XrdSecProtocolhost::getCredentials(XrdSecParameters *parm,
                                                      XrdOucErrInfo    *einfo)
{XrdSecCredentials *cp = new XrdSecCredentials;

 cp->size = 5; cp->buffer = (char *)"host";
 return cp;
}

/******************************************************************************/
/*                X r d S e c P r o t o c o l h o s t I n i t                 */
/******************************************************************************/
  
// This is a builtin protocol so we don't define an Init method. Anyway, this
// protocol need not be initialized. It works as is.

/******************************************************************************/
/*              X r d S e c P r o t o c o l h o s t O b j e c t               */
/******************************************************************************/
  
// Normally this would be defined as an extern "C", however, this function is
// statically linked into the shared library as a native protocol so there is
// no reason to define it as such. Imitators, beware! Read the comments in
// XrdSecInterface.hh
//
XrdSecProtocol *XrdSecProtocolhostObject(const char              who,
                                         const char             *hostname,
                                         const struct sockaddr  &netaddr,
                                         const char             *parms,
                                         XrdOucErrInfo          *einfo)
{

// Simply return an instance of the host protocol object
//
   return new XrdSecProtocolhost(hostname);
}
