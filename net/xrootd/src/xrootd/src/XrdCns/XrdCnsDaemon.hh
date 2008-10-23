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

#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPthread.hh"
  
class XrdCnsEvent;

class XrdCnsDaemon
{
public:

int   Configure(int argc, char **argv);

void  doRequests();

void  getEvents(XrdOucStream &);

      XrdCnsDaemon();
     ~XrdCnsDaemon() {}

private:

void  do_Create(XrdCnsEvent *evP);
void  do_Mkdir(XrdCnsEvent *evP);
void  do_Mv(XrdCnsEvent *evP);
void  do_Rm(XrdCnsEvent *evP);
void  do_Rmdir(XrdCnsEvent *evP);
void  do_Trunc(XrdCnsEvent *evP);

XrdOucStream    stdinEvents;    // STDIN fed events
XrdOucStream    fifoEvents;     // FIFO  fed events
char           *myName;
};
#endif
