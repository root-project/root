#ifndef __CMS_CLIENT__
#define __CMS_CLIENT__
/******************************************************************************/
/*                                                                            */
/*                       X r d C m s C l i e n t . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

class  XrdOucEnv;
class  XrdOucErrInfo;
class  XrdOucLogger;
struct XrdSfsPrep;

// The following return conventions are use by Forward(), Locate(), & Prepare()
//
// Return Val   Resp.errcode          Resp.errtext
// ---------    -------------------   --------
// -EREMOTE     port (0 for default)  Host name
// -EINPROGRESS n/a                   n/a
// -EEXIST      Length of errtext     Data to be returned to client as response
// > 0          Wait time (= retval)  Reason for wait
// < 0          Error number          Error message
// = 0          Not applicable        Not applicable (see below)
//                                    Forward() -> Request forwarded
//                                    Locate()  -> Redirection does not apply
//                                    Prepare() -> Request submitted
//

class XrdCmsClient
{
public:
virtual void   Added(const char *path, int Pend=0) = 0;

virtual int    Configure(char *cfn) = 0;

virtual int    Forward(XrdOucErrInfo &Resp,   const char *cmd,
                       const char    *arg1=0, const char *arg2=0,
                       const char    *arg3=0, const char *arg4=0) = 0;

virtual int    isRemote() {return myPersona == XrdCmsClient::amRemote;}

virtual int    Locate(XrdOucErrInfo &Resp, const char *path, int flags,
                      XrdOucEnv  *Info=0) = 0;

virtual int    Prepare(XrdOucErrInfo &Resp, XrdSfsPrep &pargs) = 0;

virtual void   Removed(const char *path) = 0;

virtual int    Space(XrdOucErrInfo &Resp, const char *path) = 0;

        enum   Persona {amLocal, amProxy, amRemote, amTarget};

               XrdCmsClient(Persona acting) {myPersona = acting;}
virtual       ~XrdCmsClient() {}

protected:

Persona        myPersona;
};
#endif
