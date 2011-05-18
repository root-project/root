/******************************************************************************/
/*                                                                            */
/*                    X r d N e t C m s N o t i f y . c c                     */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

#include "XrdNet/XrdNetCmsNotify.hh"
#include "XrdNet/XrdNetMsg.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysTimer.hh"

const char *XrdNetCmsNotifyCVSID = "$Id$";

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdNetCmsNotify::XrdNetCmsNotify(XrdSysError *erp, const char *aPath, 
                                 const char *iName, int Opts)
{
   char buff[1024], *p;

// Make sure we are not getting anon as an instance name
//
   if (iName) iName = XrdOucUtils::InstName(iName,0);

// Construct the path for notification
//
   p = XrdOucUtils::genPath(aPath, iName, ".olb");
   strcpy(buff, p); strcat(buff, (Opts & isServ ? "olbd.notes":"olbd.seton"));
   destPath = strdup(buff); free(p);

// Construct message object
//
   xMsg = new XrdNetMsg(erp, destPath);

// Complete initialization
//
   eDest= erp;
   Pace = !(Opts & noPace);
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdNetCmsNotify::~XrdNetCmsNotify()
{
   if (destPath) free(destPath);
   if (xMsg) delete xMsg;
}

/******************************************************************************/
/*                                  G o n e                                   */
/******************************************************************************/
  
int XrdNetCmsNotify::Gone(const char *Path, int isPfn)
{
   static const int   Cln = 6;
          const char *Cmd = (isPfn ? "gone  " : "rmdid ");
   char theMsg[MAXPATHLEN+8];
   int n;

// Construct message to be sent
//
   if ((n = strlen(Path)) > MAXPATHLEN) return -ENAMETOOLONG;
   strcpy(theMsg, Cmd); strcpy(theMsg+Cln, Path);
   n += Cln; theMsg[n] = '\n';

// Send the message
//
   return Send(theMsg, n);
}

/******************************************************************************/
/*                                  H a v e                                   */
/******************************************************************************/
  
int XrdNetCmsNotify::Have(const char *Path, int isPfn)
{
   static const int   Cln = 6;
          const char *Cmd = (isPfn ? "have  " : "newfn ");
   char theMsg[MAXPATHLEN+8];
   int n;

// Construct message to be sent
//
   if ((n = strlen(Path)) > MAXPATHLEN) return -ENAMETOOLONG;
   strcpy(theMsg, Cmd); strcpy(theMsg+Cln, Path);
   n += Cln; theMsg[n] = '\n';

// Send the message
//
   return Send(theMsg, n);
}

/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/
  
int XrdNetCmsNotify::Send(const char *theMsg, int theLen)
{
   static XrdSysMutex myMutex;
   int rc;

// Check if pacing is in effect
//
   if (Pace) {myMutex.Lock(); XrdSysTimer::Wait(10); myMutex.UnLock();}

// Send the message
//
   if ((rc = xMsg->Send(theMsg, theLen))) return rc < 0 ? rc : -ETIMEDOUT;
   return 0;
}
