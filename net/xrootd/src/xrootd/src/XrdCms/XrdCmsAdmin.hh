#ifndef __CMSADMIN__
#define __CMSADMIN__
/******************************************************************************/
/*                                                                            */
/*                        X r d C m s A d m i n . h h                         */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <stdlib.h>
#include <sys/uio.h>

#include "XrdCms/XrdCmsProtocol.hh"
#include "XrdCms/XrdCmsRRData.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdNetSocket;

class XrdCmsAdmin
{
public:

       void  Login(int socknum);

static void  setSync(XrdSysSemaphore  *sync)  {SyncUp = sync;}

       void *Notes(XrdNetSocket *AdminSock);

static void  Relay(int setSock, int newSock);

       void  Send(const char *Req, XrdCmsRRData &Data);

       void *Start(XrdNetSocket *AdminSock);

      XrdCmsAdmin() {Sname = 0; Stype = "Server"; Primary = 0;}
     ~XrdCmsAdmin() {if (Sname) free(Sname);}

private:

int   do_Login();
void  do_RmDid(int dotrim=0);
void  do_RmDud(int dotrim=0);

static XrdSysMutex      myMutex;
static XrdSysSemaphore *SyncUp;
static int              POnline;
       XrdOucStream     Stream;
       const char      *Stype;
       char            *Sname;
       int              Primary;
};
#endif
