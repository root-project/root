#ifndef _XRDOLBXMIREQ_H_
#define _XRDOLBXMIREQ_H_
/******************************************************************************/
/*                                                                            */
/*                       X r d O l b X m i R e q . h h                        */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include "XrdOlb/XrdOlbReq.hh"
#include "XrdOlb/XrdOlbXmi.hh"
#include "XrdSys/XrdSysPthread.hh"
  
class XrdOlbXmiReq : XrdOlbXmi
{
public:

       enum ReqType {do_chmod, do_mkdir, do_mkpath,do_mv,
                     do_prep,  do_rm,    do_rmdir, do_stage, do_stat};

       int  Chmod (      XrdOlbReq      *Request,
                   const char           *path,
                         mode_t          mode)
                  {return Qit(Request, do_chmod, path, (int)mode);}

       int  Mkdir (      XrdOlbReq      *Request,
                   const char           *path,
                         mode_t          mode)
                  {return Qit(Request, do_mkdir, path, (int)mode);}

       int  Mkpath(      XrdOlbReq      *Request,
                   const char           *path,
                         mode_t          mode)
                  {return Qit(Request, do_mkpath, path, (int)mode);}

       int  Prep  (const char           *ReqID,
                   const char           *path,
                         int             opts);

       int  Rename(      XrdOlbReq      *Request,
                   const char           *oldpath,
                   const char           *newpath);

       int  Remdir(      XrdOlbReq      *Request,
                   const char           *path)
                  {return Qit(Request, do_rmdir, path, 0);}

       int  Remove(      XrdOlbReq      *Request,
                   const char           *path)
                  {return Qit(Request, do_rm, path, 0);}

       int  Select(      XrdOlbReq      *Request,
                   const char           *path,
                         int             opts)
                  {return Qit(Request, do_stage, path, opts);}

       int  Stat  (      XrdOlbReq      *Request,
                   const char           *path)
                  {return Qit(Request, do_stat, path, 0);}

static void processPrpQ();

static void processReqQ();

static void processStgQ();

            XrdOlbXmiReq(XrdOlbXmi *xp);

            XrdOlbXmiReq(XrdOlbReq *reqp, ReqType rqtype, 
                      const char *path, int parms);

virtual    ~XrdOlbXmiReq();

private:
void Start();
int  Qit(XrdOlbReq *rp, ReqType, const char *path, int parms);

static XrdOlbXmi      *XmiP;
static XrdSysMutex     prpMutex;
static XrdSysSemaphore prpReady;
static XrdOlbXmiReq   *prpFirst;
static XrdOlbXmiReq   *prpLast;
static XrdSysMutex     reqMutex;
static XrdSysSemaphore reqReady;
static XrdOlbXmiReq   *reqFirst;
static XrdOlbXmiReq   *reqLast;
static XrdSysMutex     stgMutex;
static XrdSysSemaphore stgReady;
static XrdOlbXmiReq   *stgFirst;
static XrdOlbXmiReq   *stgLast;
       XrdOlbXmiReq   *Next;
       XrdOlbReq      *ReqP;
       char           *Path;
       int             Parms;
       ReqType         Rtype;
};
#endif
