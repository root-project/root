#ifndef _CMS_TALK_H
#define _CMS_TALK_H
/******************************************************************************/
/*                                                                            */
/*                         X r d C m s T a l k . h h                          */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XProtocol/YProtocol.hh"

class XrdLink;

class XrdCmsTalk
{
public:

static const char *Attend(  XrdLink *Link, XrdCms::CmsRRHdr  &Hdr,
                           char *buff, int blen,
                           int  &rlen, int tmo=5000);

static int         Complain(XrdLink *Link, int ecode, const char *msg);

static const char *Request( XrdLink *Link, XrdCms::CmsRRHdr  &Hdr,
                            char *buff, int blen);

static const char *Respond( XrdLink *Link, XrdCms::CmsRspCode rcode,
                            char *buff, int blen);

                   XrdCmsTalk() {}
                  ~XrdCmsTalk() {}

};
#endif
