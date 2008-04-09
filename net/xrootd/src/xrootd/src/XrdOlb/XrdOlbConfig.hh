#ifndef _OLB_CONFIG_H_
#define _OLB_CONFIG_H_
/******************************************************************************/
/*                                                                            */
/*                       X r d O l d C o n f i g . h h                        */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

#include <sys/socket.h>

#include "Xrd/XrdJob.hh"
#include "XrdOlb/XrdOlbPList.hh"
#include "XrdOlb/XrdOlbTypes.hh"
#include "XrdOuc/XrdOucPList.hh"
#include "XrdOuc/XrdOucTList.hh"
  
class XrdScheduler;
class XrdInet;
class XrdNetSecurity;
class XrdNetSocket;
class XrdNetWork;
class XrdSysError;
class XrdOucName2Name;
class XrdOucProg;
class XrdOucStream;
class XrdOlbXmi;

class XrdOlbConfig : public XrdJob
{
public:

int   Configure1(int argc, char **argv, char *cfn);
int   Configure2();
int   ConfigXeq(char *var, XrdOucStream &CFile, XrdSysError *eDest);
void  DoIt();
int   GenLocalPath(const char *oldp, char *newp);
int   GenMsgID(char *oldmid, char *buff, int blen);
int   inSuspend();
int   inNoStage();
int   asManager() {return isManager;}
int   asPeer()    {return isPeer;}
int   asProxy()   {return isProxy;}
int   asServer()  {return isServer;}
int   asSolo()    {return isSolo;}

int         LUPDelay;     // Maximum delay at look-up
int         LUPHold;      // Maximum hold  at look-up (in millisconds)
int         DRPDelay;     // Maximum delay for dropping an offline server
int         PSDelay;      // Maximum delay time before peer is selected
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

int         PortTCP;      // TCP Port to listen on
XrdInet    *NetTCP;       // -> Network Object

int         P_cpu;        // % CPU Capacity in load factor
int         P_fuzz;       // %     Capacity to fuzz when comparing
int         P_io;         // % I/O Capacity in load factor
int         P_load;       // % MSC Capacity in load factor
int         P_mem;        // % MEM Capacity in load factor
int         P_pag;        // % PAG Capacity in load factor

long long   DiskMin;      // Minimum KB needed of space in a partition
long long   DiskHWM;      // Minimum KB needed of space to requalify
int         DiskLinger;   // Manager Only
int         DiskAsk;      // Seconds between disk space reclaculations
int         DiskWT;       // Seconds to defer client while waiting for space
int         DiskSS;       // This is a staging server
int         PrepOK;       // Prepare processing configured

int         sched_RR;     // 1 -> Simply do round robin scheduling
int         doWait;       // 1 -> Wait for a data end-point
int         Disabled;     // 1 -> Delay director requests

XrdOucName2Name *xeq_N2N; // Server or Manager (non-null if library loaded)
XrdOucName2Name *lcl_N2N; // Server Only

char        *N2N_Lib;     // Server Only
char        *N2N_Parms;   // Server Only
char        *LocalRoot;   // Server Only
char        *RemotRoot;   // Manager
char        *MsgGID;
int          MsgGIDL;
const char  *myProg;
const char  *myName;
const char  *myDomain;
const char  *myInsName;
const char  *myInstance;
const char  *mySID;
XrdOucTList *myManagers;  // From depricated subscribe directive
XrdOucTList *ManList;     // From preferred  manager   directive

char        *NoStageFile;
char        *SuspendFile;

XrdOucProg  *ProgCH;      // Server only chmod
XrdOucProg  *ProgMD;      // Server only mkdir
XrdOucProg  *ProgMP;      // Server only mkpath
XrdOucProg  *ProgMV;      // Server only mv
XrdOucProg  *ProgRD;      // Server only rmdir
XrdOucProg  *ProgRM;      // Server only rm

unsigned long long DirFlags;
XrdOlbPList_Anchor PathList;
XrdOucPListAnchor  PexpList;
XrdNetSocket      *AdminSock;
XrdNetSocket      *AnoteSock;
XrdNetSocket      *RedirSock;
XrdNetSecurity    *Police;
struct sockaddr    myAddr;

      XrdOlbConfig() : XrdJob("olbd startup") {ConfigDefaults();}
     ~XrdOlbConfig() {}

private:

void ConfigDefaults(void);
int  ConfigN2N(void);
int  ConfigProc(int getrole=0);
int  isExec(XrdSysError *eDest, const char *ptype, char *prog);
int  MergeP(void);
int  PidFile(void);
int  setupManager(void);
int  setupServer(void);
int  setupXmi(void);
void Usage(int rc);
int  xapath(XrdSysError *edest, XrdOucStream &CFile);
int  xallow(XrdSysError *edest, XrdOucStream &CFile);
int  xcache(XrdSysError *edest, XrdOucStream &CFile);
int  Fsysadd(XrdSysError *edest, int chk, char *fn);
int  xdelay(XrdSysError *edest, XrdOucStream &CFile);
int  xdefs(XrdSysError *edest, XrdOucStream &CFile);
int  xexpo(XrdSysError *edest, XrdOucStream &CFile);
int  xfsxq(XrdSysError *edest, XrdOucStream &CFile);
int  xfxhld(XrdSysError *edest, XrdOucStream &CFile);
int  xlclrt(XrdSysError *edest, XrdOucStream &CFile);
int  xmang(XrdSysError *edest, XrdOucStream &CFile);
int  xnml(XrdSysError *edest, XrdOucStream &CFile);
int  xpath(XrdSysError *edest, XrdOucStream &CFile);
int  xperf(XrdSysError *edest, XrdOucStream &CFile);
int  xpidf(XrdSysError *edest, XrdOucStream &CFile);
int  xping(XrdSysError *edest, XrdOucStream &CFile);
int  xport(XrdSysError *edest, XrdOucStream &CFile);
int  xprep(XrdSysError *edest, XrdOucStream &CFile);
int  xprepm(XrdSysError *edest, XrdOucStream &CFile);
int  xrmtrt(XrdSysError *edest, XrdOucStream &CFile);
int  xrole(XrdSysError *edest, XrdOucStream &CFile);
int  xsched(XrdSysError *edest, XrdOucStream &CFile);
int  xspace(XrdSysError *edest, XrdOucStream &CFile);
int  xsubs(XrdSysError *edest, XrdOucStream &CFile);
int  xtrace(XrdSysError *edest, XrdOucStream &CFile);
int  xxmi(XrdSysError *edest, XrdOucStream &CFile);

XrdNetWork       *NetTCPr;     // Network for supervisors
XrdOucTList      *monPath;     // cache directive paths
XrdOucTList      *monPathP;    // path  directive paths (w or s only)
char             *AdminPath;
int               AdminMode;
char             *pidPath;
char             *ConfigFN;
char            **inArgv;
int               inArgc;
char             *XmiPath;
char             *XmiParms;
int               isManager;
int               isPeer;
int               isProxy;
int               isServer;
int               isSolo;
char             *myRole;
char             *perfpgm;
int               perfint;
int               cachelife;
int               pendplife;
};

namespace XrdOlb
{
extern    XrdScheduler *Sched;
extern    XrdOlbConfig  Config;
extern    XrdOlbXmi    *Xmi_Chmod;
extern    XrdOlbXmi    *Xmi_Load;
extern    XrdOlbXmi    *Xmi_Mkdir;
extern    XrdOlbXmi    *Xmi_Mkpath;
extern    XrdOlbXmi    *Xmi_Prep;
extern    XrdOlbXmi    *Xmi_Rename;
extern    XrdOlbXmi    *Xmi_Remdir;
extern    XrdOlbXmi    *Xmi_Remove;
extern    XrdOlbXmi    *Xmi_Select;
extern    XrdOlbXmi    *Xmi_Space;
extern    XrdOlbXmi    *Xmi_Stat;
}
#endif
