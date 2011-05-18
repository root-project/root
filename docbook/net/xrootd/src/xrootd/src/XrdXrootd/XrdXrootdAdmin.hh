#ifndef __XROOTDADMIN__
#define __XROOTDADMIN__
/******************************************************************************/
/*                                                                            */
/*                     X r d X r o o t d A d m i n . h h                      */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <stdlib.h>
#include <netinet/in.h>

#include "Xrd/XrdLinkMatch.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XProtocol/XProtocol.hh"

class XrdNetSocket;
class XrdXrootdJob;

class XrdXrootdAdmin
{
public:

static void  addJob(const char *jname, XrdXrootdJob *jp);

static int   Init(XrdSysError *erp, XrdNetSocket *asock);

       void  Login(int socknum);

       void *Start(XrdNetSocket *AdminSock);

       XrdXrootdAdmin() {}
      ~XrdXrootdAdmin() {}

private:
int   do_Abort();
int   do_Cj();
int   do_Cont();
int   do_Disc();
int   do_Login();
int   do_Lsc();
int   do_Lsj();
int   do_Lsj_Xeq(XrdXrootdJob *jp);
int   do_Lsd();
int   do_Msg();
int   do_Pause();
int   do_Red();
char *getMsg(char *msg, int &mlen);
int   getreqID();
int   getTarget(const char *act, char **rest=0);
int   sendErr(int rc, const char *act, const char *msg);
int   sendOK(int sent);
int   sendResp(const char *act, XActionCode anum);
int   sendResp(const char *act, XActionCode anum,
               const char *msg, int mlen);
void  Xeq();

struct JobTable {struct JobTable *Next;
                 char            *Jname;
                 XrdXrootdJob    *Job;
                };

static JobTable        *JobList;

static XrdSysError     *eDest;
       XrdOucStream     Stream;
       XrdLinkMatch     Target;

struct usr {kXR_unt16   pad;
            kXR_unt16   atn;
            kXR_int32   len;
            kXR_int32   act;
            usr() {pad = 0; atn = htons(kXR_attn);}
           ~usr() {}
           }            usResp;
       char             TraceID[24];
       char             reqID[16];
};
#endif
