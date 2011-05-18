#ifndef __SEC_PMANAGER_HH__
#define __SEC_PMANAGER_HH__
/******************************************************************************/
/*                                                                            */
/*                     X r d S e c P M a n a g e r . h h                      */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include <sys/socket.h>
  
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdOucErrInfo;
class XrdSecProtList;
class XrdSecProtocol;

typedef int XrdSecPMask_t;

#define PROTPARMS const char, const char *, const struct sockaddr &, \
                  const char *, XrdOucErrInfo *

class XrdSecPManager
{
public:

XrdSecPMask_t   Find(const         char  *pid,      // In
                                   char **parg=0);  // Out

XrdSecProtocol *Get(const char     *hname,
                    const sockaddr &netaddr,
                    const char     *pname,
                    XrdOucErrInfo  *erp);

XrdSecProtocol *Get (const char             *hname,
                     const struct sockaddr  &netaddr,
                           XrdSecParameters &secparm);

int             Load(XrdOucErrInfo *eMsg,    // In
                     const char     pmode,   // In 'c' | 's'
                     const char    *pid,     // In
                     const char    *parg,    // In
                     const char    *path)    // In
                     {return (0 != ldPO(eMsg, pmode, pid, parg, path));}

void            setDebug(int dbg) {DebugON = dbg;}

                XrdSecPManager(int dbg=0)
                   {First = Last = 0; DebugON = dbg; protnum = 1;}
               ~XrdSecPManager() {}

private:

XrdSecProtList    *Add(XrdOucErrInfo  *eMsg, const char *pid,
                       XrdSecProtocol *(*ep)(PROTPARMS), const char *parg);
XrdSecProtList    *ldPO(XrdOucErrInfo *eMsg,    // In
                        const char     pmode,   // In 'c' | 's'
                        const char    *pid,     // In
                        const char    *parg=0,  // In
                        const char    *spath=0);// In
XrdSecProtList    *Lookup(const char *pid);

XrdSecPMask_t      protnum;
XrdSysMutex        myMutex;
XrdSecProtList    *First;
XrdSecProtList    *Last;
int                DebugON;
};
#endif
