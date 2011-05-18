#ifndef _CMS_SECURITY_H
#define _CMS_SECURITY_H
/******************************************************************************/
/*                                                                            */
/*                     X r d C m s S e c u r i t y . h h                      */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$


#include <sys/types.h>
#include <sys/socket.h>

#include "XrdSec/XrdSecInterface.hh"

class XrdLink;
class XrdOucTList;

class XrdCmsSecurity
{
public:

static int             Authenticate(XrdLink *Link, const char *Token, int tlen);

static int             Configure(const char *Lib, const char *Cfn=0);

static const char     *getToken(int &size, const char *hostname);

static int             Identify(XrdLink *Link, XrdCms::CmsRRHdr &inHdr,
                                char *authBuff, int abLen);

static char           *setSystemID(XrdOucTList *tp, const char *iName,
                                   const char  *iHost,    char  iType);

      XrdCmsSecurity() {}
     ~XrdCmsSecurity() {}

private:
static XrdSecService *DHS;
};
#endif
