#ifndef __XRDCnsLog_H_
#define __XRDCnsLog_H_
/******************************************************************************/
/*                                                                            */
/*                          X r d C n s L o g . h h                           */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include "XrdOuc/XrdOucNSWalk.hh"

class XrdOucTList;

class XrdCnsLog
{
public:

static XrdOucTList         *Dirs(const char *Path, int &rc);

static XrdOucNSWalk::NSEnt *List(const char *logDir,
                                 XrdOucNSWalk::NSEnt **Base,
                                 int isEP=0);

static const char          *invFNa;  // Name of lcl inventory log file phase 1
static const char          *invFNt;  // Name of lcl inventory log file phase 2
static const char          *invFNz;  // Name of rmt inventory log file

               XrdCnsLog() {}
              ~XrdCnsLog() {}

private:
static int isEP(const char *Path);
};
#endif
