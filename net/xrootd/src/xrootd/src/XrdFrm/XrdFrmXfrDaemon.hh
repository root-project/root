#ifndef __FRMXFRDAEMON_H__
#define __FRMXFRDAEMON_H__
/******************************************************************************/
/*                                                                            */
/*                    X r d F r m X f r D a e m o n . h h                     */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

#include "XrdFrm/XrdFrmReqBoss.hh"

class XrdFrmXfrDaemon
{
public:

static int  Init();

static void Pong();

static int  Start();

           XrdFrmXfrDaemon() {}
          ~XrdFrmXfrDaemon() {}

private:
static XrdFrmReqBoss *Boss(char bType);

static XrdFrmReqBoss GetBoss;
static XrdFrmReqBoss PutBoss;
static XrdFrmReqBoss MigBoss;
static XrdFrmReqBoss StgBoss;
};
#endif
