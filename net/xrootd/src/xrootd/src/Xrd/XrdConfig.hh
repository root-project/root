#ifndef _XRD_CONFIG_H
#define _XRD_CONFIG_H
/******************************************************************************/
/*                                                                            */
/*                          X r d C o n f i g . h h                           */
/*                                                                            */
/* (C) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//          $Id$ 

#include "Xrd/XrdProtocol.hh"

class XrdNetSecurity;
class XrdOucStream;
class XrdConfigProt;

class XrdConfig
{
public:

int   Configure(int argc, char **argv);

int   ConfigXeq(char *var, XrdOucStream &Config, XrdSysError *eDest=0);

      XrdConfig();
     ~XrdConfig() {}

private:

int   ASocket(const char *path, const char *fname, mode_t mode);
int   ConfigProc(void);
int   getUG(char *parm, uid_t &theUid, gid_t &theGid);
int   setFDL();
int   Setup(char *dfltp);
void  Usage(int rc);
int   xallow(XrdSysError *edest, XrdOucStream &Config);
int   xapath(XrdSysError *edest, XrdOucStream &Config);
int   xbuf(XrdSysError *edest, XrdOucStream &Config);
int   xnet(XrdSysError *edest, XrdOucStream &Config);
int   xlog(XrdSysError *edest, XrdOucStream &Config);
int   xport(XrdSysError *edest, XrdOucStream &Config);
int   xprot(XrdSysError *edest, XrdOucStream &Config);
int   xrep(XrdSysError *edest, XrdOucStream &Config);
int   xsched(XrdSysError *edest, XrdOucStream &Config);
int   xtrace(XrdSysError *edest, XrdOucStream &Config);
int   xtmo(XrdSysError *edest, XrdOucStream &Config);
int   yport(XrdSysError *edest, const char *ptyp, const char *pval);

static const char  *TraceID;

XrdProtocol_Config  ProtInfo;
XrdNetSecurity     *Police;
const char         *myProg;
const char         *myName;
const char         *myDomain;
const char         *myInsName;
char               *myInstance;
char               *AdminPath;
char               *ConfigFN;
char               *repDest[2];
XrdConfigProt      *Firstcp;
XrdConfigProt      *Lastcp;
int                 Net_Blen;
int                 Net_Opts;
int                 Wan_Blen;
int                 Wan_Opts;

int                 PortTCP;      // TCP Port to listen on
int                 PortUDP;      // UDP Port to listen on (currently unsupported)
int                 PortWAN;      // TCP port to listen on for WAN connections
int                 AdminMode;
int                 repInt;
char                repOpts;
char                isProxy;
};
#endif
