#ifndef __OLBADMIN__
#define __OLBADMIN__
/******************************************************************************/
/*                                                                            */
/*                        X r d O l b A d m i n . h h                         */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <stdlib.h>

#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPthread.hh"

class XrdNetSocket;

class XrdOlbAdmin
{
public:

       void  Login(int socknum);

static void  setSync(XrdSysSemaphore  *sync)  {SyncUp = sync;}

       void *Notes(XrdNetSocket *AdminSock);

       void *Start(XrdNetSocket *AdminSock);

      XrdOlbAdmin() {Sname = 0; Stype = "Server"; Primary = 0;}
     ~XrdOlbAdmin() {if (Sname) free(Sname);}

private:

int   do_Login();
void  do_NoStage();
void  do_Resume();
void  do_RmDid(int dotrim=0);
void  do_RmDud(int dotrim=0);
void  do_Stage();
void  do_Suspend();

static XrdSysMutex      myMutex;
static XrdSysSemaphore *SyncUp;
static int              POnline;
       XrdOucStream     Stream;
       const char      *Stype;
       char            *Sname;
       int              Primary;
};
#endif
