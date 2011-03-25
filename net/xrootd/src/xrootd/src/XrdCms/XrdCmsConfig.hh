#ifndef _CMS_CONFIG_H_
#define _CMS_CONFIG_H_
/******************************************************************************/
/*                                                                            */
/*                       X r d C m s C o n f i g . h h                        */
/*                                                                            */
/* (c) 2011 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <stdlib.h>
#include <sys/socket.h>

#include "Xrd/XrdJob.hh"
#include "XrdCms/XrdCmsPList.hh"
#include "XrdCms/XrdCmsTypes.hh"
#include "XrdOuc/XrdOucPList.hh"
#include "XrdOuc/XrdOucTList.hh"
  
class XrdInet;
class XrdScheduler;
class XrdNetSecurity;
class XrdNetSocket;
class XrdOss;
class XrdSysError;
class XrdOucName2Name;
class XrdOucProg;
class XrdOucStream;
class XrdCmsAdmin;
class XrdCmsXmi;

class XrdCmsConfig : public XrdJob
{
public:

int   Configure1(int argc, char **argv, char *cfn);
int   Configure2();
int   ConfigXeq(char *var, XrdOucStream &CFile, XrdSysError *eDest);
void  DoIt();
int   GenLocalPath(const char *oldp, char *newp);
int   asManager() {return isManager;}
int   asPeer()    {return isPeer;}
int   asProxy()   {return isProxy;}
int   asServer()  {return isServer;}
int   asSolo()    {return isSolo;}

int         LUPDelay;     // Maximum delay at look-up
int         LUPHold;      // Maximum hold  at look-up (in millisconds)
int         DRPDelay;     // Maximum delay for dropping an offline server
int         PSDelay;      // Maximum delay time before peer is selected
int         RWDelay;      // R/W lookup delay handling (0 | 1 | 2)
int         QryDelay;     // Query Response Deadline
int         SRVDelay;     // Minimum delay at startup
int         SUPCount;     // Minimum server count
int         SUPLevel;     // Minimum server count as floating percentage
int         SUPDelay;     // Maximum delay when server count falls below min
int         SUSDelay;     // Maximum delay when suspended
int         MaxLoad;      // Maximum load
int         MaxDelay;     // Maximum load delay
int         MsgTTL;       // Maximum msg lifetime
int         RefReset;     // Min seconds    before a global ref count reset
int         RefTurn;      // Min references before a global ref count reset
int         AskPerf;      // Seconds between perf queries
int         AskPing;      // Number of ping requests per AskPerf window
int         LogPerf;      // AskPerf intervals before logging perf

int         PortTCP;      // TCP Port to  listen on
XrdInet    *NetTCP;       // -> Network Object

int         P_cpu;        // % CPU Capacity in load factor
int         P_dsk;        // % DSK Capacity in load factor
int         P_fuzz;       // %     Capacity to fuzz when comparing
int         P_io;         // % I/O Capacity in load factor
int         P_load;       // % MSC Capacity in load factor
int         P_mem;        // % MEM Capacity in load factor
int         P_pag;        // % PAG Capacity in load factor

int         DiskMin;      // Minimum MB needed of space in a partition
int         DiskHWM;      // Minimum MB needed of space to requalify
short       DiskMinP;     // Minimum MB needed of space in a partition as %
short       DiskHWMP;     // Minimum MB needed of space to requalify   as %
int         DiskLinger;   // Manager Only
int         DiskAsk;      // Seconds between disk space reclaculations
int         DiskWT;       // Seconds to defer client while waiting for space
int         DiskSS;       // This is a staging server
int         DiskOK;       // This configuration has data

int         sched_RR;     // 1 -> Simply do round robin scheduling
int         doWait;       // 1 -> Wait for a data end-point

XrdOucName2Name *xeq_N2N; // Server or Manager (non-null if library loaded)
XrdOucName2Name *lcl_N2N; // Server Only

char        *ossLib;      // -> oss library
char        *ossParms;    // -> oss library parameters
char        *N2N_Lib;     // Server Only
char        *N2N_Parms;   // Server Only
char        *LocalRoot;   // Server Only
char        *RemotRoot;   // Manager
char        *myPaths;     // Exported paths
const char  *myProg;
const char  *myName;
const char  *myDomain;
const char  *myInsName;
const char  *myInstance;
const char  *mySID;
XrdOucTList *ManList;     // From manager directive
XrdOucTList *NanList;     // From manager directive (managers only)

XrdOss      *ossFS;       // The filsesystem interface
XrdOucProg  *ProgCH;      // Server only chmod
XrdOucProg  *ProgMD;      // Server only mkdir
XrdOucProg  *ProgMP;      // Server only mkpath
XrdOucProg  *ProgMV;      // Server only mv
XrdOucProg  *ProgRD;      // Server only rmdir
XrdOucProg  *ProgRM;      // Server only rm
XrdOucProg  *ProgTR;      // Server only trunc

unsigned long long DirFlags;
XrdCmsPList_Anchor PathList;
XrdOucPListAnchor  PexpList;
XrdNetSocket      *AdminSock;
XrdNetSocket      *AnoteSock;
XrdNetSocket      *RedirSock;
XrdNetSecurity    *Police;
struct sockaddr    myAddr;

      XrdCmsConfig() : XrdJob("cmsd startup") {ConfigDefaults();}
     ~XrdCmsConfig() {}

private:

void ConfigDefaults(void);
int  ConfigN2N(void);
int  ConfigOSS(void);
int  ConfigProc(int getrole=0);
int  isExec(XrdSysError *eDest, const char *ptype, char *prog);
int  MergeP(void);
int  PidFile(void);
int  setupManager(void);
int  setupServer(void);
char *setupSid();
int  setupXmi(void);
void Usage(int rc);
int  xapath(XrdSysError *edest, XrdOucStream &CFile);
int  xallow(XrdSysError *edest, XrdOucStream &CFile);
int  Fsysadd(XrdSysError *edest, int chk, char *fn);
int  xdelay(XrdSysError *edest, XrdOucStream &CFile);
int  xdefs(XrdSysError *edest, XrdOucStream &CFile);
int  xdfs(XrdSysError *edest, XrdOucStream &CFile);
int  xexpo(XrdSysError *edest, XrdOucStream &CFile);
int  xfsxq(XrdSysError *edest, XrdOucStream &CFile);
int  xfxhld(XrdSysError *edest, XrdOucStream &CFile);
int  xlclrt(XrdSysError *edest, XrdOucStream &CFile);
int  xmang(XrdSysError *edest, XrdOucStream &CFile);
int  xnml(XrdSysError *edest, XrdOucStream &CFile);
int  xolib(XrdSysError *edest, XrdOucStream &CFile);
int  xperf(XrdSysError *edest, XrdOucStream &CFile);
int  xpidf(XrdSysError *edest, XrdOucStream &CFile);
int  xping(XrdSysError *edest, XrdOucStream &CFile);
int  xprep(XrdSysError *edest, XrdOucStream &CFile);
int  xprepm(XrdSysError *edest, XrdOucStream &CFile);
int  xrmtrt(XrdSysError *edest, XrdOucStream &CFile);
int  xrole(XrdSysError *edest, XrdOucStream &CFile);
int  xsched(XrdSysError *edest, XrdOucStream &CFile);
int  xsecl(XrdSysError *edest, XrdOucStream &CFile);
int  xspace(XrdSysError *edest, XrdOucStream &CFile);
int  xtrace(XrdSysError *edest, XrdOucStream &CFile);
int  xxmi(XrdSysError *edest, XrdOucStream &CFile);

XrdInet          *NetTCPr;     // Network for supervisors
char             *AdminPath;
int               AdminMode;
char             *pidPath;
char             *ConfigFN;
char            **inArgv;
int               inArgc;
char             *SecLib;
char             *XmiPath;
char             *XmiParms;
int               isManager;
int               isMeta;
int               isPeer;
int               isProxy;
int               isServer;
int               isSolo;
char             *myRole;
char             *perfpgm;
int               perfint;
int               cachelife;
int               pendplife;
int               FSlim;
};
namespace XrdCms
{
extern XrdCmsAdmin   Admin;
extern XrdCmsConfig  Config;
extern XrdScheduler *Sched;
extern XrdCmsXmi    *Xmi_Chmod;
extern XrdCmsXmi    *Xmi_Load;
extern XrdCmsXmi    *Xmi_Mkdir;
extern XrdCmsXmi    *Xmi_Mkpath;
extern XrdCmsXmi    *Xmi_Prep;
extern XrdCmsXmi    *Xmi_Rename;
extern XrdCmsXmi    *Xmi_Remdir;
extern XrdCmsXmi    *Xmi_Remove;
extern XrdCmsXmi    *Xmi_Select;
extern XrdCmsXmi    *Xmi_Space;
extern XrdCmsXmi    *Xmi_Stat;
}
#endif
