#ifndef __CMS_LOGIN_H__
#define __CMS_LOGIN_H__
/******************************************************************************/
/*                                                                            */
/*                        X r d C m s L o g i n . h h                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include <sys/uio.h>

#include "XProtocol/XPtypes.hh"
#include "XProtocol/YProtocol.hh"

class XrdLink;

class XrdCmsLogin
{
public:

       int  Admit(XrdLink *Link, XrdCms::CmsLoginData &Data);

static int  Login(XrdLink *Link, XrdCms::CmsLoginData &Data, int timeout=-1);

       XrdCmsLogin(char *Buff = 0, int Blen = 0) {myBuff = Buff; myBlen = Blen;}

      ~XrdCmsLogin() {}

private:

static int Authenticate(XrdLink *Link, XrdCms::CmsLoginData &Data);
static int Emsg(XrdLink *, const char *, int ecode=XrdCms::kYR_EINVAL);
static int sendData(XrdLink *Link, XrdCms::CmsLoginData &Data);

         char       *myBuff;
         int         myBlen;
};
#endif
