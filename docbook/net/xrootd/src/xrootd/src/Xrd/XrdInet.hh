#ifndef __XRD_INET_H__
#define __XRD_INET_H__
/******************************************************************************/
/*                                                                            */
/*                            X r d I n e t . h h                             */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <unistd.h>

#include "XrdNet/XrdNet.hh"

// The XrdInet class defines a generic network where we can define common
// initial tcp/ip and udp operations. It is based on the generalized network
// support framework. However, Accept and Connect have been augmented to
// provide for more scalable communications handling.
//
class XrdSysError;
class XrdNetSecurity;
class XrdLink;

class XrdInet : public XrdNet
{
public:

XrdLink    *Accept(int opts=0, int timeout=-1);

XrdLink    *Connect(const char *host, int port, int opts=0, int timeout=-1);

            XrdInet(XrdSysError *erp, XrdNetSecurity *secp=0)
                      : XrdNet(erp, secp) {}
           ~XrdInet() {}
private:

static const char *TraceID;
};
#endif
