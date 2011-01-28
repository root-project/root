#ifndef __SEC_PROTOCOL_HOST_H__
#define __SEC_PROTOCOL_HOST_H__
/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l h o s t . h h                  */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include <stdlib.h>
#include <strings.h>

#include "XrdSec/XrdSecInterface.hh"

class XrdSecProtocolhost : public XrdSecProtocol
{
public:

        int                Authenticate  (XrdSecCredentials  *cred,
                                          XrdSecParameters  **parms,
                                          XrdOucErrInfo      *einfo=0);

        XrdSecCredentials *getCredentials(XrdSecParameters  *parm=0,
                                          XrdOucErrInfo     *einfo=0);

        const char        *getParms(int &psize, const char *hname=0)
                                   {psize = 5; return "host";}


void                       Delete() {delete this;}

              XrdSecProtocolhost(const char *host) {theHost = strdup(host);}
             ~XrdSecProtocolhost() {if (theHost) free(theHost);}
private:

char *theHost;
};
#endif
