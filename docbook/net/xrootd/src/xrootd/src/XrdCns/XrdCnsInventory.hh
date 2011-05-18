#ifndef __XRDCnsInventory_H_
#define __XRDCnsInventory_H_
/******************************************************************************/
/*                                                                            */
/*                    X r d C n s I n v e n t o r y . h h                     */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <sys/param.h>

#include "XrdCns/XrdCnsLogRec.hh"
#include "XrdCns/XrdCnsLogFile.hh"
#include "XrdCns/XrdCnsXref.hh"
#include "XrdOuc/XrdOucNSWalk.hh"

class XrdCnsInventory
{
public:

int   Conduct(const char *dPath);

int   Init(XrdCnsLogFile *theLF);

      XrdCnsInventory();
     ~XrdCnsInventory() {}

private:
int          Xref(XrdOucNSWalk::NSEnt *nP);

XrdCnsLogRec   dRec;
XrdCnsLogRec   fRec;
XrdCnsLogRec   mRec;
XrdCnsLogRec   sRec;
XrdCnsXref     Mount;
XrdCnsXref     Space;
XrdCnsLogFile *lfP;
char           lfnBuff[MAXPATHLEN+1];
const char    *cwdP;
char           mDflt;
char           sDflt;
};
#endif
