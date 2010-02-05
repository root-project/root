#ifndef _CNS_DAEMON_H_
#define _CNS_DAEMON_H_
/******************************************************************************/
/*                                                                            */
/*                       X r d C n s D a e m o n . h h                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include "XrdSys/XrdSysPthread.hh"

class XrdOucStream;

class XrdCnsDaemon
{
public:

void  getEvents(XrdOucStream &, const char *Who);

      XrdCnsDaemon() {}
     ~XrdCnsDaemon() {}

private:

char *getLFN(XrdOucStream &Events);
};
#endif
