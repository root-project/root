#ifndef __XRDSECSERVER_H__
#define __XRDSECSERVER_H__
/******************************************************************************/
/*                                                                            */
/*                       X r d S e c S e r v e r . h h                        */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSec/XrdSecPManager.hh"

class XrdSecProtBind;
class XrdOucTrace;
  
class XrdSecServer : XrdSecService
{
public:

const char             *getParms(int &size, const char *hname=0);

// = 0 -> No protocol can be returned (einfo has the reason)
// ! 0 -> Address of protocol object is bing returned.
//
XrdSecProtocol         *getProtocol(const char              *host,    // In
                                    const struct sockaddr   &hadr,    // In
                                    const XrdSecCredentials *cred,    // In
                                    XrdOucErrInfo           *einfo=0);// Out

int                     Configure(const char *cfn);

                        XrdSecServer(XrdSysLogger *lp);
                       ~XrdSecServer() {}      // Server is never deleted

private:

static XrdSecPManager  PManager;

XrdSysError     eDest;
XrdOucTrace    *SecTrace;
XrdSecProtBind *bpFirst;
XrdSecProtBind *bpLast;
XrdSecProtBind *bpDefault;
char           *SToken;
char           *STBuff;
int             STBlen;
int             Enforce;
int             implauth;

int             add2token(XrdSysError &erp,char *,char **,int &,XrdSecPMask_t &);
int             ConfigFile(const char *cfn);
int             ConfigXeq(char *var, XrdOucStream &Config, XrdSysError &Eroute);
int             ProtBind_Complete(XrdSysError &Eroute);
int             xpbind(XrdOucStream &Config, XrdSysError &Eroute);
int             xpparm(XrdOucStream &Config, XrdSysError &Eroute);
int             xprot(XrdOucStream &Config, XrdSysError &Eroute);
int             xtrace(XrdOucStream &Config, XrdSysError &Eroute);
};
#endif
