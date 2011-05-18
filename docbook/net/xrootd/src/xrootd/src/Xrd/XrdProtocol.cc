/******************************************************************************/
/*                                                                            */
/*                        X r d P r o t o c o l . c c                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$  

const char *XrdProtocolCVSID = "$Id$";

#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>

#include "Xrd/XrdProtocol.hh"
  
/******************************************************************************/
/*   X r d P r o t o c o l _ C o n f i g   C o p y   C o n s t r u c t o r    */
/******************************************************************************/
  
XrdProtocol_Config::XrdProtocol_Config(XrdProtocol_Config &rhs)
{
eDest     = rhs.eDest;
NetTCP    = rhs.NetTCP;
BPool     = rhs.BPool;
Sched     = rhs.Sched;
Stats     = rhs.Stats;
Threads   = rhs.Threads;
Trace     = rhs.Trace;

ConfigFN  = rhs.ConfigFN ? strdup(rhs.ConfigFN) : 0;
Format    = rhs.Format;
Port      = rhs.Port;
WSize     = rhs.WSize;
AdmPath   = rhs.AdmPath  ? strdup(rhs.AdmPath)  : 0;
AdmMode   = rhs.AdmMode;
myInst    = rhs.myInst   ? strdup(rhs.myInst)   : 0;
myName    = rhs.myName   ? strdup(rhs.myName)   : 0;
         if (!rhs.myAddr) myAddr = 0;
            else {myAddr = (struct sockaddr *)malloc(sizeof(struct sockaddr));
                  memcpy(myAddr, rhs.myAddr, sizeof(struct sockaddr));
                 }
ConnMax   = rhs.ConnMax;
readWait  = rhs.readWait;
idleWait  = rhs.idleWait;
hailWait  = rhs.hailWait;
argc      = rhs.argc;
argv      = rhs.argv;
DebugON   = rhs.DebugON;
WANPort   = rhs.WANPort;
WANWSize  = rhs.WANWSize;
}
