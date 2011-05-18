#ifndef __CMS_FINDER__
#define __CMS_FINDER__
/******************************************************************************/
/*                                                                            */
/*                       X r d C m s F i n d e r . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XrdCms/XrdCmsClient.hh"

#include "XrdSys/XrdSysPthread.hh"

class  XrdCmsClientMan;
class  XrdOss;
class  XrdOucEnv;
class  XrdOucErrInfo;
class  XrdOucTList;
struct XrdCmsData;
class  XrdCmsRRData;
struct XrdSfsPrep;
class  XrdSysLogger;

/******************************************************************************/
/*                         R e m o t e   F i n d e r                          */
/******************************************************************************/

class XrdCmsFinderRMT : public XrdCmsClient
{
public:
        void   Added(const char *path, int Pend=0) {}

        int    Configure(char *cfn);

        int    Forward(XrdOucErrInfo &Resp, const char *cmd,
                       const char *arg1=0,  const char *arg2=0,
                       const char *arg3=0,  const char *arg4=0);

        int    Locate(XrdOucErrInfo &Resp, const char *path, int flags,
                      XrdOucEnv *Info=0);

        int    Prepare(XrdOucErrInfo &Resp, XrdSfsPrep &pargs);

        void   Removed(const char *path) {}

        void   setSS(XrdOss *thess) {}

        int    Space(XrdOucErrInfo &Resp, const char *path);

               XrdCmsFinderRMT(XrdSysLogger *lp, int whoami=0, int Port=0);
              ~XrdCmsFinderRMT();

static const int MaxMan = 16;

private:
int              Decode(char **resp);
void             Inform(XrdCmsClientMan *xman, struct iovec xmsg[], int xnum);
XrdCmsClientMan *SelectManager(XrdOucErrInfo &Resp, const char *path);
void             SelectManFail(XrdOucErrInfo &Resp);
int              send2Man(XrdOucErrInfo &, const char *, struct iovec *, int);
int              StartManagers(XrdOucTList *);

XrdCmsClientMan *myManTable[MaxMan];
XrdCmsClientMan *myManagers;
int              myManCount;
XrdSysMutex      myData;
char            *CMSPath;
int              ConWait;
int              RepDelay;
int              RepNone;
int              RepWait;
int              PrepWait;
int              isMeta;
int              isTarget;
int              myPort;
unsigned char    SMode;
unsigned char    sendID;
};

/******************************************************************************/
/*                         T a r g e t   F i n d e r                          */
/******************************************************************************/

class XrdOucStream;
  
class XrdCmsFinderTRG : public XrdCmsClient
{
public:
        void   Added(const char *path, int Pend=0);

        int    Configure(char *cfn);

        int    Forward(XrdOucErrInfo &Resp,   const char *cmd,
                       const char    *arg1=0, const char *arg2=0,
                       const char    *arg3=0, const char *arg4=0) {return 0;}

        int    Locate(XrdOucErrInfo &Resp, const char *path, int flags,
                      XrdOucEnv *Info=0) {return 0;}

        int    Prepare(XrdOucErrInfo &Resp, XrdSfsPrep &pargs) {return 0;}

        void   Removed(const char *path);

        int    RunAdmin(char *Path);

        int    Space(XrdOucErrInfo &Resp, const char *path) {return 0;}

        void  *Start();

               XrdCmsFinderTRG(XrdSysLogger *, int, int, XrdOss *theSS=0);
              ~XrdCmsFinderTRG();

private:

void  Hookup();
int   Process(XrdCmsRRData &Data);

XrdOss        *SS;
XrdOucStream  *CMSp;
XrdSysMutex    myData;
int            myPort;
char          *CMSPath;
char          *Login;
int            isRedir;
int            isProxy;
int            Active;
};

namespace XrdCms
{
enum  {IsProxy = 1, IsRedir = 2, IsTarget = 4, IsMeta = 8};
}
#endif
