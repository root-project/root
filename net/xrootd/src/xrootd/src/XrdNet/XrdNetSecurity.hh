#ifndef __NETSECURITY__
#define __NETSECURITY__
/******************************************************************************/
/*                                                                            */
/*                     X r d N e t S e c u r i t y . h h                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

#include <ctype.h>
#include <stdlib.h>
  
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucNList.hh"
#include "XrdSys/XrdSysPthread.hh"

class  XrdNetTextList;
class  XrdOucTrace;
struct sockaddr;

class XrdNetSecurity
{
public:
  
void  AddHost(char *hname);

void  AddNetGroup(char *hname);

char *Authorize(struct sockaddr *addr);

void  Merge(XrdNetSecurity *srcp);  // Deletes srcp

void  Trace(XrdOucTrace *et=0) {eTrace = et;}

     XrdNetSecurity() {NetGroups = 0; eTrace = 0; lifetime = 8*60*60;}
    ~XrdNetSecurity() {}

private:

char *hostOK(char *hname, const char *ipname, const char *why);

XrdOucNList_Anchor        HostList;

XrdNetTextList           *NetGroups;

XrdOucHash<char>          OKHosts;
XrdSysMutex               okHMutex;
XrdOucTrace              *eTrace;

int                       lifetime;
static const char        *TraceID;
};
#endif
