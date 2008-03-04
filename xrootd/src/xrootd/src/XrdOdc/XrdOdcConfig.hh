#ifndef _ODC_CONFIG_H
#define _ODC_CONFIG_H
/******************************************************************************/
/*                                                                            */
/*                       X r d O d c C o n f i g . h h                        */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//          $Id$

#include "XrdOdc/XrdOdcConfDefs.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOuca2x.hh"
  
class XrdSysError;
class XrdOucStream;

/******************************************************************************/
/*                    X r d O d c C o n f i g   C l a s s                     */
/******************************************************************************/

#define ODC_FAILOVER 'f'
#define ODC_ROUNDROB 'r'
  
class XrdOdcConfig
{
public:

int           Configure(char *cfn, const char *mode, int isBoth=0);

int           ConWait;      // Seconds to wait for a manager connection
int           RepWait;      // Seconds to wait for manager replies
int           RepWaitMS;    // RepWait*1000 for poll()
int           RepDelay;     // Seconds to delay before retrying manager
int           RepNone;      // Max number of consecutive non-responses
int           PrepWait;     // Millisecond wait between prepare requests

char         *OLBPath;      // Path to the local olb for target nodes
char         *myHost;
const char   *myName;

XrdOucTList  *ManList;      // List of managers for remote redirection
XrdOucTList  *PanList;      // List of managers for proxy  redirection
unsigned char SMode;        // Manager selection mode
unsigned char SModeP;       // Manager selection mode (proxy)

      XrdOdcConfig(XrdSysError *erp)
                  {ConWait = 10; RepWait = 6; RepWaitMS = 3000; RepDelay = 5;
                   PrepWait = 33; ManList = PanList = 0;
                   SMode = SModeP = ODC_FAILOVER;
                   eDest = erp;
                   OLBPath = 0; RepNone = 8;
                  }
     ~XrdOdcConfig();

private:
int ConfigProc(char *cfn);
int ConfigXeq(char *var, XrdOucStream &Config);
int xapath(XrdSysError *eDest, XrdOucStream &Config);
int xconw(XrdSysError *eDest, XrdOucStream &Config);
int xmang(XrdSysError *eDest, XrdOucStream &Config);
int xreqs(XrdSysError *eDest, XrdOucStream &Config);
int xtrac(XrdSysError *eDest, XrdOucStream &Config);

XrdSysError   *eDest;
};
#endif
