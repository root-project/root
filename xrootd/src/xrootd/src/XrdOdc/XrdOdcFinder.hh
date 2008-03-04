#ifndef __ODC_FINDER__
#define __ODC_FINDER__
/******************************************************************************/
/*                                                                            */
/*                       X r d O d c F i n d e r . h h                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XrdOdc/XrdOdcConfDefs.hh"
#include "XrdSys/XrdSysPthread.hh"

class  XrdOdcManager;
class  XrdOucEnv;
class  XrdSysError;
class  XrdOucErrInfo;
class  XrdSysLogger;
class  XrdOucTList;
struct XrdOdcData;
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

class XrdOdcFinder
{
public:
virtual int    Configure(char *cfn) = 0;

virtual int    Forward(XrdOucErrInfo &Resp, const char *cmd, 
                       const char *arg1=0, const char *arg2=0) = 0;

virtual int    isRemote() {return myPersona == XrdOdcFinder::amRemote;}

virtual int    Locate(XrdOucErrInfo &Resp, const char *path, int flags,
                      XrdOucEnv *Info=0) = 0;

virtual int    Prepare(XrdOucErrInfo &Resp, XrdSfsPrep &pargs) = 0;

        enum   Persona {amLocal, amProxy, amRemote, amTarget};

               XrdOdcFinder(XrdSysLogger *lp, Persona acting);
virtual       ~XrdOdcFinder() {}

protected:

Persona myPersona;
static char *OLBPath;
};

/******************************************************************************/
/*                         R e m o t e   F i n d e r                          */
/******************************************************************************/

#define XRDODCMAXMAN 16

#define XrdOdcIsProxy  1
#define XrdOdcIsRedir  2
#define XrdOdcIsTarget 4
  
class XrdOdcFinderRMT : public XrdOdcFinder
{
public:
        int    Configure(char *cfn);

        int    Forward(XrdOucErrInfo &Resp, const char *cmd, 
                       const char *arg1=0, const char *arg2=0);

        int    Locate(XrdOucErrInfo &Resp, const char *path, int flags,
                      XrdOucEnv *Info=0);

        int    Prepare(XrdOucErrInfo &Resp, XrdSfsPrep &pargs);

               XrdOdcFinderRMT(XrdSysLogger *lp, int whoami = 0);
              ~XrdOdcFinderRMT();

private:
int            Decode(char **resp);
XrdOdcManager *SelectManager(XrdOucErrInfo &Resp, const char *path);
void           SelectManFail(XrdOucErrInfo &Resp);
int            send2Man(XrdOucErrInfo &, const char *, struct iovec *, int);
int            StartManagers(XrdOucTList *);

XrdOdcManager *myManTable[XRDODCMAXMAN];
XrdOdcManager *myManagers;
int            myManCount;
XrdSysMutex    myData;
int            ConWait;
int            RepDelay;
int            RepNone;
int            RepWait;
int            PrepWait;
int            isTarget;
unsigned char  SMode;
};

/******************************************************************************/
/*                         T a r g e t   F i n d e r                          */
/******************************************************************************/

class XrdOucStream;
  
class XrdOdcFinderTRG : public XrdOdcFinder
{
public:
        void   Added(const char *path);

        int    Configure(char *cfn);

        int    Forward(XrdOucErrInfo &Resp, const char *cmd,
                       const char *arg1=0, const char *arg2=0) {return 0;}

        int    Locate(XrdOucErrInfo &Resp, const char *path, int flags,
                      XrdOucEnv *Info=0) {return 0;}

        int    Prepare(XrdOucErrInfo &Resp, XrdSfsPrep &pargs) {return 0;}

        void   Removed(const char *path);

        void  *Start();

               XrdOdcFinderTRG(XrdSysLogger *lp, int whoami, int port);
              ~XrdOdcFinderTRG();

private:

void  Hookup();

XrdOucStream  *OLBp;
XrdSysMutex    myData;
int            myPort;
char          *OLBPath;
char          *Login;
int            isRedir;
int            isProxy;
int            Active;
};
#endif
