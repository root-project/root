#ifndef _CMS_CLIENTCONFIG_H
#define _CMS_CLIENTCONFIG_H
/******************************************************************************/
/*                                                                            */
/*                 X r d C m s C l i e n t C o n f i g . h h                  */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOuca2x.hh"
  
class XrdOucStream;
class XrdSysError;

#define ODC_FAILOVER 'f'
#define ODC_ROUNDROB 'r'
  
class XrdCmsClientConfig
{
public:

enum configHow  {configMeta   = 1, configNorm  = 2, configProxy  = 4};
enum configWhat {configMan    = 1, configSuper = 2, configServer = 4};

int           Configure(char *cfn, configWhat What, configHow How);

int           ConWait;      // Seconds to wait for a manager connection
int           RepWait;      // Seconds to wait for manager replies
int           RepWaitMS;    // RepWait*1000 for poll()
int           RepDelay;     // Seconds to delay before retrying manager
int           RepNone;      // Max number of consecutive non-responses
int           PrepWait;     // Millisecond wait between prepare requests
int           FwdWait;      // Millisecond wait between foward  requests
int           haveMeta;     // Have a meta manager (only if we are a manager)

char         *CMSPath;      // Path to the local cmsd for target nodes
char         *myHost;
const char   *myName;

XrdOucTList  *ManList;      // List of managers for remote redirection
XrdOucTList  *PanList;      // List of managers for proxy  redirection
unsigned char SMode;        // Manager selection mode
unsigned char SModeP;       // Manager selection mode (proxy)

enum {FailOver = 'f', RoundRob = 'r'};

      XrdCmsClientConfig() : ConWait(10), RepWait(3),  RepWaitMS(3000),
                             RepDelay(5), RepNone(8),  PrepWait(33),
                             FwdWait(0),  haveMeta(0), CMSPath(0),
                             myHost(0),   myName(0),
                             ManList(0),  PanList(0),
                             SMode(FailOver), SModeP(FailOver), isMeta(0) {}
     ~XrdCmsClientConfig();

private:
int isMeta;   // We are  a meta manager
int isMan;    // We are  a      manager

int ConfigProc(char *cfn);
int ConfigXeq(char *var, XrdOucStream &Config);
int xapath(XrdOucStream &Config);
int xconw(XrdOucStream  &Config);
int xmang(XrdOucStream  &Config);
int xreqs(XrdOucStream  &Config);
int xtrac(XrdOucStream  &Config);
};
#endif
