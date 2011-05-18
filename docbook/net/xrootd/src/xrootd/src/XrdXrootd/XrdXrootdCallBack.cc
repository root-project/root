/******************************************************************************/
/*                                                                            */
/*                  X r d X r o o t d C a l l B a c k . c c                   */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

const char *XrdXrootdCallBackCVSID = "$Id$";

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/uio.h>

#include "Xrd/XrdScheduler.hh"
#include "XProtocol/XProtocol.hh"
#include "XProtocol/XPtypes.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdXrootd/XrdXrootdCallBack.hh"
#include "XrdXrootd/XrdXrootdProtocol.hh"
#include "XrdXrootd/XrdXrootdStats.hh"
#include "XrdXrootd/XrdXrootdReqID.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
class XrdXrootdCBJob : XrdJob
{
public:

static XrdXrootdCBJob *Alloc(XrdXrootdCallBack *cbF, XrdOucErrInfo *erp, int rval);

       void            DoIt();

inline void            Recycle(){myMutex.Lock();
                                 Next = FreeJob;
                                 FreeJob = this;
                                 myMutex.UnLock();
                                }

                       XrdXrootdCBJob(XrdXrootdCallBack *cbp,
                                      XrdOucErrInfo     *erp,
                                      int                rval)
                                     : XrdJob("async response"),
                                       cbFunc(cbp), eInfo(erp), Result(rval) {}
                      ~XrdXrootdCBJob() {}

private:
void DoStatx(XrdOucErrInfo *eInfo);
static XrdSysMutex         myMutex;
static XrdXrootdCBJob     *FreeJob;

XrdXrootdCBJob            *Next;
XrdXrootdCallBack         *cbFunc;
XrdOucErrInfo             *eInfo;
int                        Result;
};

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

extern XrdOucTrace       *XrdXrootdTrace;

       XrdSysError       *XrdXrootdCallBack::eDest;
       XrdXrootdStats    *XrdXrootdCallBack::SI;
       XrdScheduler      *XrdXrootdCallBack::Sched;
       int                XrdXrootdCallBack::Port;

       XrdSysMutex        XrdXrootdCBJob::myMutex;
       XrdXrootdCBJob    *XrdXrootdCBJob::FreeJob;

/******************************************************************************/
/*                        X r d X r o o t d C B J o b                         */
/******************************************************************************/
/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdXrootdCBJob *XrdXrootdCBJob::Alloc(XrdXrootdCallBack *cbF,
                                      XrdOucErrInfo     *erp,
                                      int                rval)
{
   XrdXrootdCBJob *cbj;

// Obtain a call back object by trying to avoid new()
//
   myMutex.Lock();
   if (!(cbj = FreeJob)) cbj = new XrdXrootdCBJob(cbF, erp, rval);
      else {cbj->cbFunc = cbF, cbj->eInfo = erp; 
            cbj->Result = rval;FreeJob = cbj->Next;
           }
   myMutex.UnLock();

// Return the new object
//
   return cbj;
}

/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/
  
void XrdXrootdCBJob::DoIt()
{

// Some operations differ in  the way we handle them. For instance, for open()
// if it succeeds then we must force the client to retry the open request
// because we can't attach the file to the client here. We do this by asking
// the client to wait zero seconds. Protocol demands a client retry.
//
   if (SFS_OK == Result)
      {if (*(cbFunc->Func()) == 'o') cbFunc->sendResp(eInfo, kXR_wait, 0);
          else {if (*(cbFunc->Func()) == 'x') DoStatx(eInfo);
                cbFunc->sendResp(eInfo, kXR_ok, 0, eInfo->getErrText());
               }
      } else cbFunc->sendError(Result, eInfo);

// Tell the requestor that the callback has completed
//
   if (eInfo->getErrCB()) eInfo->getErrCB()->Done(Result, eInfo);
      else delete eInfo;
   eInfo = 0;
   Recycle();
}
  
/******************************************************************************/
/*                               D o S t a t x                                */
/******************************************************************************/
  
void XrdXrootdCBJob::DoStatx(XrdOucErrInfo *einfo)
{
   const char *tp = einfo->getErrText();
   char cflags[2];
   int flags;

// Skip to the third token
//
   while(*tp && *tp == ' ') tp++;
   while(*tp && *tp != ' ') tp++; // 1st
   while(*tp && *tp == ' ') tp++;
   while(*tp && *tp != ' ') tp++; // 2nd

// Convert to flags
//
   flags = atoi(tp);

// Convert to proper indicator
//
        if (flags & kXR_offline) cflags[0] = (char)kXR_offline;
   else if (flags & kXR_isDir)   cflags[0] = (char)kXR_isDir;
   else                          cflags[0] = (char)kXR_file;

// Set the new response
//
   cflags[1] = '\0';
   einfo->setErrInfo(0, cflags);
}

/******************************************************************************/
/*                     X r d X r o o t d C a l l B a c k                      */
/******************************************************************************/
/******************************************************************************/
/*                                  D o n e                                   */
/******************************************************************************/
  
void XrdXrootdCallBack::Done(int           &Result,   //I/O: Function result
                             XrdOucErrInfo *eInfo)    // In: Error information
{
   XrdXrootdCBJob *cbj;

// Sending an async response may take a long time. So, we schedule the task
// to run asynchronously from the forces that got us here.
//
   if (!(cbj = XrdXrootdCBJob::Alloc(this, eInfo, Result)))
      {eDest->Emsg("Done",ENOMEM,"get call back job; user",eInfo->getErrUser());
       if (eInfo->getErrCB()) eInfo->getErrCB()->Done(Result, eInfo);
          else delete eInfo;
      } else Sched->Schedule((XrdJob *)cbj);
}

/******************************************************************************/
/*                                  S a m e                                   */
/******************************************************************************/
  
int XrdXrootdCallBack::Same(unsigned long long arg1, unsigned long long arg2)
{
   XrdXrootdReqID ReqID1(arg1), ReqID2(arg2);
   unsigned char sid1[2], sid2[2];
   unsigned int  inst1, inst2;
            int  lid1, lid2;

   ReqID1.getID(sid1, lid1, inst1);
   ReqID2.getID(sid2, lid2, inst2);
   return lid1 == lid2;
}

/******************************************************************************/
/*                             s e n d E r r o r                              */
/******************************************************************************/
  
void XrdXrootdCallBack::sendError(int            rc,
                                  XrdOucErrInfo *eInfo)
{
   const char *TraceID = "fsError";
   static int Xserr = kXR_ServerError;
   int ecode;
   const char *eMsg = eInfo->getErrText(ecode);
   const char *User = eInfo->getErrUser();

// Optimize error message handling here
//
   if (eMsg && !*eMsg) eMsg = 0;

// Process standard errors
//
   if (rc == SFS_ERROR)
      {SI->errorCnt++;
       rc = XrdXrootdProtocol::mapError(ecode);
       sendResp(eInfo, kXR_error, &rc, eMsg, 1);
       return;
      }

// Process the redirection (error msg is host:port)
//
   if (rc == SFS_REDIRECT)
      {SI->redirCnt++;
       if (ecode <= 0) ecode = (ecode ? -ecode : Port);
       TRACE(REDIR, User <<" async redir to " << eMsg <<':' <<ecode);
       sendResp(eInfo, kXR_redirect, &ecode, eMsg);
       return;
      }

// Process the deferal
//
   if (rc >= SFS_STALL)
      {SI->stallCnt++;
       TRACE(STALL, "Stalling " <<User <<" for " <<rc <<" sec");
       sendResp(eInfo, kXR_wait, &rc, eMsg, 1);
       return;
      }

// Process the data response
//
   if (rc == SFS_DATA)
      {if (ecode) sendResp(eInfo, kXR_ok, 0, eMsg, ecode);
         else     sendResp(eInfo, kXR_ok, 0);
       return;
      }

// Unknown conditions, report it
//
   {char buff[32];
    SI->errorCnt++;
    sprintf(buff, "%d", rc);
    eDest->Emsg("sendError", "Unknown error code", buff, eMsg);
    sendResp(eInfo, kXR_error, &Xserr, eMsg, 1);
    return;
   }
}

/******************************************************************************/
/*                              s e n d R e s p                               */
/******************************************************************************/
  
void XrdXrootdCallBack::sendResp(XrdOucErrInfo  *eInfo,
                                 XResponseType   Status,
                                 int            *Data,
                                 const char     *Msg,
                                 int             ovhd)
{
   const char *TraceID = "sendResp";
   struct iovec       rspVec[4];
   XrdXrootdReqID     ReqID;
   int                dlen = 0, n = 1;
   kXR_int32          xbuf;

   if (Data)
      {xbuf = static_cast<kXR_int32>(htonl(*Data));
               rspVec[n].iov_base = (caddr_t)(&xbuf);
       dlen  = rspVec[n].iov_len  = sizeof(xbuf); n++;           // 1
      }
    if (Msg && *Msg)
       {        rspVec[n].iov_base = (caddr_t)Msg;
        dlen += rspVec[n].iov_len  = strlen(Msg)+ovhd; n++;      // 2
       }

// Set the destination
//
   ReqID.setID(eInfo->getErrArg());

// Send the async response
//
   if (XrdXrootdResponse::Send(ReqID, Status, rspVec, n, dlen) < 0)
      eDest->Emsg("sendResp", eInfo->getErrUser(), Opname, 
                  "async resp aborted; user gone.");
      else if (TRACING(TRACE_RSP))
              {XrdXrootdResponse theResp;
               theResp.Set(ReqID.Stream());
               TRACE(RSP, eInfo->getErrUser() <<" async " <<theResp.ID()
                          <<' ' <<Opname <<" status " <<Status);
              }
}
