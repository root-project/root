/******************************************************************************/
/*                                                                            */
/*                       X r d C m s X m i R e q . c c                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCmsXmiReqCVSID = "$Id$";

#include <string.h>

#include "XrdCms/XrdCmsTrace.hh"
#include "XrdCms/XrdCmsXmiReq.hh"
  
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdCmsXmi       *XrdCmsXmiReq::XmiP;
XrdSysMutex      XrdCmsXmiReq::prpMutex;
XrdSysSemaphore  XrdCmsXmiReq::prpReady(0);
XrdCmsXmiReq    *XrdCmsXmiReq::prpFirst = 0;
XrdCmsXmiReq    *XrdCmsXmiReq::prpLast  = 0;
XrdSysMutex      XrdCmsXmiReq::reqMutex;
XrdSysSemaphore  XrdCmsXmiReq::reqReady(0);
XrdCmsXmiReq    *XrdCmsXmiReq::reqFirst = 0;
XrdCmsXmiReq    *XrdCmsXmiReq::reqLast  = 0;
XrdSysMutex      XrdCmsXmiReq::stgMutex;
XrdSysSemaphore  XrdCmsXmiReq::stgReady(0);
XrdCmsXmiReq    *XrdCmsXmiReq::stgFirst = 0;
XrdCmsXmiReq    *XrdCmsXmiReq::stgLast  = 0;

using namespace XrdCms;

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *XrdCmsXmi_StartPrpQ(void *parg)
{  
   XrdCmsXmiReq *requestProcessor = (XrdCmsXmiReq *)parg;

   requestProcessor->processPrpQ();

   return (void *)0;
}
  
void *XrdCmsXmi_StartReqQ(void *parg)
{  
   XrdCmsXmiReq *requestProcessor = (XrdCmsXmiReq *)parg;

   requestProcessor->processReqQ();

   return (void *)0;
}
  
void *XrdCmsXmi_StartStgQ(void *parg)
{  
   XrdCmsXmiReq *requestProcessor = (XrdCmsXmiReq *)parg;

   requestProcessor->processStgQ();

   return (void *)0;
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdCmsXmiReq::XrdCmsXmiReq(XrdCmsXmi *xp)
{
   ReqP    = 0;
   Path    = 0;
   Parms   = 0;
   Next    = 0;
   XmiP    = xp;
   Start();
}
  
XrdCmsXmiReq::XrdCmsXmiReq(XrdCmsReq *reqp, ReqType rtype, int parms,
                           const char *path,  const char *opaque,
                           const char *path2, const char *opaque2)
{

   ReqP    = reqp;
   Parms   = parms;
   Path    = strdup(path);
   Opaque  = (opaque  ? strdup(opaque)  : 0);
   Path2   = (path2   ? strdup(path2)   : 0);
   Opaque2 = (opaque2 ? strdup(opaque2) : 0);
   Rtype   = rtype;
   Next    = 0;

// Place ourselves on the proper request queue
//
        if (rtype == do_stage)
           {stgMutex.Lock();
            if (stgLast) {stgLast->Next = this; stgLast = this;}
               else      {stgFirst = stgLast = this; stgReady.Post();}
            stgMutex.UnLock();
           }
   else if (rtype == do_prep)
           {prpMutex.Lock();
            if (prpLast) {prpLast->Next = this; prpLast = this;}
               else      {prpFirst = prpLast = this; prpReady.Post();}
            prpMutex.UnLock();
           }
   else    {reqMutex.Lock();
            if (reqLast) {reqLast->Next = this; reqLast = this;}
               else      {reqFirst = reqLast = this; reqReady.Post();}
            reqMutex.UnLock();
           }
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdCmsXmiReq::~XrdCmsXmiReq()
{
   if (Path)    free(Path);
   if (Opaque)  free(Opaque);
   if (Path2)   free(Path2);
   if (Opaque2) free(Opaque2);
   if (ReqP) delete ReqP;
}

/******************************************************************************/
/*                           p r o c e s s P r p Q                            */
/******************************************************************************/
  
void XrdCmsXmiReq::processPrpQ()
{
   XrdCmsXmiReq *myQueue, *rp;

// This is one big loop where we take off as many requests from the queue
// as we can. However we feed them one at a time the Xmi prep   processor
// as we have found that the interfaces can be so gruesome that batching
// requests outweighs incurs complexity beyond belief. For prepare, no
// responses are possible, so we pass a null XmiReq pointer.
//
   while(1)
        {prpReady.Wait();
         prpMutex.Lock();
         myQueue  = prpFirst;
         prpFirst = prpLast = 0;
         prpMutex.UnLock();

         while((rp = myQueue))
              {myQueue = rp->Next;
               XmiP->Prep(rp->Path2, rp->Parms, rp->Path, rp->Opaque);
               delete rp;
              }
        }
}

/******************************************************************************/
/*                           p r o c e s s R e q Q                            */
/******************************************************************************/
  
void XrdCmsXmiReq::processReqQ()
{
   XrdCmsXmiReq *myQueue, *rp;
   int rc;

// This is one big loop where we take off as many requests from the queue
// as we can and feed them to the general request processor
//
do {reqReady.Wait();
    reqMutex.Lock();
    myQueue  = reqFirst;
    reqFirst = reqLast = 0;
    reqMutex.UnLock();

    while((rp = myQueue))
         {myQueue = rp->Next;
          switch(rp->Rtype)
                {case do_stat:   
                      rc = XmiP->Stat(rp->ReqP, rp->Path, rp->Opaque);
                      break;
                 case do_mkdir:  
                      rc = XmiP->Mkdir(rp->ReqP, rp->Parms, rp->Path, rp->Opaque);
                      break;
                 case do_mkpath: 
                      rc = XmiP->Mkpath(rp->ReqP, rp->Parms, rp->Path, rp->Opaque);
                      break;
                 case do_rmdir:  
                      rc = XmiP->Remdir(rp->ReqP, rp->Path, rp->Opaque);
                      break;
                 case do_rm:     
                      rc = XmiP->Remove(rp->ReqP, rp->Path, rp->Opaque);
                      break;
                 case do_mv:
                      rc = XmiP->Rename(rp->ReqP, rp->Path,  rp->Opaque,
                                                  rp->Path2, rp->Opaque2);
                      break;
                 case do_chmod:
                      rc = XmiP->Chmod(rp->ReqP, rp->Parms, rp->Path, rp->Opaque);
                      break;
                 default: Say.Emsg("reqQ", "Invalid request code.");
                          rp->ReqP->Reply_Error("Internal server error");
                          rc = 1;
                          break;
                }
          if (!rc) rp->ReqP->Reply_Error("Function failed in xmi handler");
          delete rp;
         }
   } while(1);
}

/******************************************************************************/
/*                           p r o c e s s S t g Q                            */
/******************************************************************************/
  
void XrdCmsXmiReq::processStgQ()
{
   XrdCmsXmiReq *myQueue, *rp;

// This is one big loop where we take off as many requests from the queue
// as we can. However we feed them one at a time the Xmi select processor
// as we have found that the interfaces can be so gruesome that batching
// requests outweighs incurs complexity beyond belief.
//
   while(1)
        {stgReady.Wait();
         stgMutex.Lock();
         myQueue  = stgFirst;
         stgFirst = stgLast = 0;
         stgMutex.UnLock();

         while((rp = myQueue))
              {myQueue = rp->Next;
               if (!XmiP->Select(rp->ReqP, rp->Parms,rp->Path, rp->Opaque))
                  rp->ReqP->Reply_Error("Select failed in xmi handler");
               delete rp;
              }
        }
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
void XrdCmsXmiReq::Start()
{
   pthread_t tid;
   int       retc;

// Start the thread that handles prepare requests
//
   if ((retc = XrdSysThread::Run(&tid, XrdCmsXmi_StartPrpQ, (void *)this,
                            XRDSYSTHREAD_BIND, "xmi prepare handler")))
      {Say.Emsg("XmiReq", retc, "create prepare thread"); _exit(3);}

// Start the thread that handles general requests
//
   if ((retc = XrdSysThread::Run(&tid, XrdCmsXmi_StartReqQ, (void *)this,
                            XRDSYSTHREAD_BIND, "xmi request handler")))
      {Say.Emsg("XmiReq", retc, "create request thread"); _exit(3);}

// Start the thread that handles staging requests
//
   if ((retc = XrdSysThread::Run(&tid, XrdCmsXmi_StartStgQ, (void *)this,
                            XRDSYSTHREAD_BIND, "xmi staging handler")))
      {Say.Emsg("XmiReq", retc, "create staging thread"); _exit(3);}
}
 
/******************************************************************************/
/*                                   Q i t                                    */
/******************************************************************************/
  
int XrdCmsXmiReq::Qit(XrdCmsReq *rp, ReqType rt, int parms,
                      const char *path,  const char *opaque,
                      const char *path2, const char *opaque2)
{
    new XrdCmsXmiReq((rp ? rp->Reply_WaitResp(5940) : 0),
                     rt, parms, path, opaque, path2, opaque2);
    return 1;
}
