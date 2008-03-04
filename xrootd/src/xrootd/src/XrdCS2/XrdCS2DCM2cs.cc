/******************************************************************************/
/*                                                                            */
/*                          X r d C S 2 D C M . c c                           */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCS2DCMCVSID = "$Id$";

#include <unistd.h>
#include <fcntl.h>
#include <iostream.h>
#include <string.h>
#include <stdio.h>
#include <utime.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include "Xrd/XrdTrace.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysTimer.hh"
#include "XrdSys/XrdSysPthread.hh"

#include "XrdCS2/XrdCS2DCM.hh"

#include "XrdNet/XrdNetLink.hh"

#include "Cns_api.h"

#include "castor/Constants.hpp"
#include "castor/Services.hpp"
#include "castor/stager/IJobSvc.hpp"
#include "castor/stager/DiskCopy.hpp"
#include "castor/IService.hpp"
#include "castor/BaseObject.hpp"
#include "castor/stager/SubRequest.hpp"
#include "castor/exception/Exception.hpp"
#include "castor/exception/Communication.hpp"
#include "h/stager_client_api.h"

/******************************************************************************/
/*           G l o b a l   C o n f i g u r a t i o n   O b j e c t            */
/******************************************************************************/


extern XrdSysError       XrdLog;

extern XrdSysLogger      XrdLogger;

extern XrdOucTrace       XrdTrace;
 
extern XrdCS2DCM         XrdCS2d;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
/******************************************************************************/
/*                       X r d C S 2 D C M H e l p e r                        */
/******************************************************************************/
  
struct XrdCS2DCMHelper
{
   struct stage_io_fileresp *Resp;
   char *Req;

   XrdCS2DCMHelper() {Resp = 0; Req = 0;}
  ~XrdCS2DCMHelper() {if (Resp) free(Resp);
                      if (Req)  free(Req);
                     }
};

/******************************************************************************/
/*                      X r d C S 2 D C M S e r v i c e                       */
/******************************************************************************/
  
class XrdCS2DCMService
{
public:

static const int maxServ = 8;

castor::stager::IJobSvc *jobSvc;

static XrdCS2DCMService *Get();

void                     Rel();

static int               Init();

                         XrdCS2DCMService() {Next = 0;}
                        ~XrdCS2DCMService() {} // Never deleted!

private:

static XrdSysSemaphore   mySem;
static XrdSysMutex       myMutex;
static XrdCS2DCMService *Free;

       XrdCS2DCMService *Next;
       castor::Services *AllSvcs;
       castor::IService *JobSvcs;
};

XrdSysSemaphore   XrdCS2DCMService::mySem(0);
XrdSysMutex       XrdCS2DCMService::myMutex;
XrdCS2DCMService *XrdCS2DCMService::Free = 0;

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdCS2DCMService::Init()
{
   XrdCS2DCMService *sp;
   int Num = maxServ;

   while(Num--)
        {sp = new XrdCS2DCMService();
         try {sp->AllSvcs = castor::BaseObject::services();
              if (!sp->AllSvcs)
                 {XrdLog.Emsg("Init", "Unable to allocate base services.");
                  return 0;
                 }
              sp->JobSvcs = sp->AllSvcs->service("RemoteJobSvc",castor::SVC_REMOTEJOBSVC);
              if (!sp->JobSvcs)
                 {XrdLog.Emsg("Init", "Unable to allocate job services.");
                  return 0;
                 }
              sp->jobSvc = dynamic_cast<castor::stager::IJobSvc*>(sp->JobSvcs);
              if (!sp->jobSvc)
                 {XrdLog.Emsg("Init", "Unable to cast JobSvcs to jobSvc.");
                  return 0;
                 }
              sp->Next = Free; Free = sp; mySem.Post();
             }  catch (castor::exception::Exception e)
                      {XrdLog.Emsg("Init","Fatal svc exception;",
                                   e.getMessage().str().c_str());
                       return 0;
                      }
        }

// All done
//
   return 1;
};

/******************************************************************************/
/*                                   G e t                                    */
/******************************************************************************/
  
XrdCS2DCMService *XrdCS2DCMService::Get()
{
   XrdCS2DCMService *sp;

// Wait until we have an available service
//
   do {mySem.Wait();
       myMutex.Lock();
       if (Free) {sp = Free; Free = sp->Next;}
       myMutex.UnLock();
      } while(!sp);

   return sp;
}

/******************************************************************************/
/*                                   R e l                                    */
/******************************************************************************/
  
void XrdCS2DCMService::Rel()
{

// Simply recycle the service
//
   myMutex.Lock();
   Next = Free;
   Free = this;
   mySem.Post();
   myMutex.UnLock();
}

/******************************************************************************/
/*                             X r d C S 2 D C M                              */
/******************************************************************************/
/******************************************************************************/
/*                              C S 2 _ I n i t                               */
/******************************************************************************/
  
int XrdCS2DCM::CS2_Init()
{
    castor::BaseObject::initLog("", castor::SVC_NOMSG);
    return XrdCS2DCMService::Init();
}

/******************************************************************************/
/*                              C S 2 _ O p e n                               */
/******************************************************************************/
  
int XrdCS2DCM::CS2_Open(const char *Tid, const char *Fid, char *Lfn,
                        int flags, off_t fsize)
{
   const char *TraceID = "_Open";
   static const int retryWait = 13*1000;
   static const char *Proto = "xroot";
   XrdCS2DCMHelper CS2;
   int rc, ecode;

// The following code was lifted from rfio
//
   do {if ((rc=stage_open(NULL, Proto, Lfn, flags, 0660, fsize,
                          &CS2.Resp, &CS2.Req, NULL)) < 0)
          {ecode = serrno;
           if ((ecode == SECOMERR)     || (ecode == SETIMEDOUT)
           ||  (ecode == ECONNREFUSED))
              {TRACE(DEBUG, "Cannot connect to Castor2!");
               XrdSysTimer::Wait(retryWait);
              }
              else {XrdLog.Emsg("Open", Lfn, "open failed;", sstrerror(ecode));
                    return 0;
                   }
          }
      } while(rc < 0);

// We should have a response here
//
   if (CS2.Resp == NULL)
      {XrdLog.Emsg("Open", Lfn, "received null response");
       return 0;
      }
          
// Check if we got some other strange error
//
   if (CS2.Resp->errorCode != 0)
      {char *eMsg, eBuff[32];
       if (CS2.Resp->errorMessage) eMsg = CS2.Resp->errorMessage;
          else {sprintf(eBuff, "error %d", CS2.Resp->errorCode); eMsg = eBuff;}
       XrdLog.Emsg("Open", Lfn, "open error;", eMsg);
       return 0;
      }

// Return success
//
   TRACE(DEBUG, "Stage_Open " <<Lfn <<" succeeded");
   return 1;
}

/******************************************************************************/
/*                             C S 2 _ r D o n e                              */
/******************************************************************************/
  
int XrdCS2DCM::CS2_rDone(const char *Tid, unsigned long long reqID, 
                         const char *Lfn)
{
  const char *TraceID = "_rDone";
  XrdCS2DCMService *sp;
  int allOK = 1;

// Obatin a service object
//
   TRACE(DEBUG, "Calling getUpdateDone(" <<reqID <<") for " <<Lfn);
   sp = XrdCS2DCMService::Get();

// Issue the done
//
  try {sp->jobSvc->getUpdateDone(reqID);}

  catch (castor::exception::Communication e)
        {XrdLog.Emsg("rDone", Tid, "Communications error;",
                                   e.getMessage().str().c_str());
         allOK = 0;
        }
  catch (castor::exception::Exception e)
        {XrdLog.Emsg("rDone",Tid, "Fatal exception;",
                                  e.getMessage().str().c_str());
         allOK = 0;;
        }

// Another message here
//
   if (!allOK) XrdLog.Emsg("rDone", Tid, "done processing failed for", Lfn);

// Release the object and return
//
   sp->Rel();
   return allOK;
}

/******************************************************************************/
/*                             C S 2 _ w D o n e                              */
/******************************************************************************/
  
int  XrdCS2DCM::CS2_wDone(const char *Tid, unsigned long long reqID, 
                          const char *Pfn)
{
  const char *TraceID = "_wDone";
  castor::stager::SubRequest subReq;
  XrdCS2DCMService *sp;
  struct stat buf;
  unsigned long long fileSize = 0;
  int allOK = 1;

// Obtain the file size
//
   if (stat(Pfn, &buf))
      {XrdLog.Emsg("wDone", errno, "stat", Pfn);
       fileSize = 0x7fffffff;
      } else fileSize = buf.st_size;

// Obatin a service object
//
   sp = XrdCS2DCMService::Get();

// Insert the requestid into the subRequest
//
   subReq.setId(reqID);

// Issue the prepare for Migration followed by a putDone (new files) or
// getUpdateDone (existing files opened for r/w).
//
  try {TRACE(DEBUG, "Calling prepareForMigration(" <<reqID <<',' <<fileSize <<") for " <<Pfn);
       sp->jobSvc->prepareForMigration(&subReq, fileSize);
      }
  catch (castor::exception::Communication e)
        {XrdLog.Emsg("wDone", Tid, "Communications error;",
                                   e.getMessage().str().c_str());
         allOK = 0;
        }
  catch (castor::exception::Exception e)
        {XrdLog.Emsg("wDone",Tid, "Fatal exception;",
                                  e.getMessage().str().c_str());
         allOK = 0;;
        }

// Another message here
//
   if (!allOK) XrdLog.Emsg("wDone", Tid, "done processing failed for", Pfn);

// Release the object and return
//
   sp->Rel();
   return allOK;
}

/******************************************************************************/
/*                             C S 2 _ w F a i l                              */
/******************************************************************************/
  
int  XrdCS2DCM::CS2_wFail(const char *Tid, unsigned long long reqID, 
                          const char *Pfn, int isNew)
{
  const char *TraceID = "_wFail";
  XrdCS2DCMService *sp;
  int allOK = 1;

// Obatin a service object
//
   sp = XrdCS2DCMService::Get();

// Issue the putFailed (new file) or getUpdateFailed (existing files open for rw)
//
  try {if (isNew)
          {TRACE(DEBUG, "Calling putFailed(" <<reqID <<") for " <<Pfn);
           sp->jobSvc->putFailed(reqID);
          } else {
           TRACE(DEBUG, "Calling getUpdateFailed(" <<reqID <<") for " <<Pfn);
           sp->jobSvc->getUpdateFailed(reqID);
          }
      }
  catch (castor::exception::Communication e)
        {XrdLog.Emsg("wFail", Tid, "Communications error;",
                                   e.getMessage().str().c_str());
         allOK = 0;
        }
  catch (castor::exception::Exception e)
        {XrdLog.Emsg("wFail",Tid, "Fatal exception;",
                                  e.getMessage().str().c_str());
         allOK = 0;;
        }

// Another message here
//
   if (!allOK) XrdLog.Emsg("wFail", Tid, "fail processing failed for", Pfn);

// Release the object and return
//
   sp->Rel();
   return allOK;
}

/******************************************************************************/
/*                               a d d L i n k                                */
/******************************************************************************/
  
void  XrdCS2DCM::addLink(const char *Rfn, const char *Lfn)
{
   char Pfn[2048];
   struct iovec iov[3];

// Convert the lfn to our pfn
//
   strcpy(Pfn, XPath);
   strcpy(Pfn+XPlen, Lfn);

// Create the symlink to the real file
//
   if (symlink(Rfn, Pfn))
      {XrdLog.Emsg("addLink", errno, "create symlink", Pfn);
       return;
      }

// Notify the olbd that we now have the file
//
   if (olbdLink)
      {iov[0].iov_base = (char *)"newfn "; iov[0].iov_len = 6;
       iov[1].iov_base = (char *)Lfn;      iov[1].iov_len = strlen(Lfn);
       iov[2].iov_base = (char *)"\n";     iov[0].iov_len = 1;
       olbdLink->Send(iov, 3);
      }
}

/******************************************************************************/
/*                               d e l L i n k                                */
/******************************************************************************/
  
void  XrdCS2DCM::delLink(const char *Lfn)
{
   char Pfn[2048];
   struct iovec iov[3];

// Convert the lfn to our pfn
//
   strcpy(Pfn, XPath);
   strcpy(Pfn+XPlen, Lfn);

// Create the symlink to the real file
//
   if (unlink(Lfn))
      {if (errno != ENOENT) XrdLog.Emsg("delLink", errno, "remove symlink", Lfn);
       return;
      }

// Notify the olbd that we no longer have the file
//
   if (olbdLink)
      {iov[0].iov_base = (char *)"rmdid "; iov[0].iov_len = 6;
       iov[1].iov_base = (char *)Lfn;      iov[1].iov_len = strlen(Lfn);
       iov[2].iov_base = (char *)"\n";     iov[0].iov_len = 1;
       olbdLink->Send(iov, 3);
      }
}
 
/******************************************************************************/
/*                                  P r e p                                   */
/******************************************************************************/

void XrdCS2DCM::Prep(const char *Fid, const char *Rfn)
{
   char *servP, *endP, Lfn[CA_MAXPATHLEN+1];
   unsigned long long reqID, Fnum;

// Extract out the requestid from Fid (<reqid>:<fid>@<server>)
//
   if ((endP = index(Fid, ':')))
      {Fid = endP+1;
       reqID = strtoll(Fid, &endP, 10);
       if (*endP != ':') reqID = 0;
      } else reqID = 0;

// Convert the Fid to the Castor file name (lfn)
//
   if ((servP = index(Fid, '@')))
      {servP++;
       Fnum = strtoll(Fid, &endP, 10);
       if (*endP != '@') servP = 0;
      }
   if (!servP) 
      {XrdLog.Emsg("Prep", Fid, "fileid invalid in", Rfn);
       return;
      }

// Convert the fid to the actual Castor file name
//
   if (Cns_getpath(servP, Fnum, Lfn) != 0)
      {XrdLog.Emsg("Prep", Fid, "fid conversion failed;", sstrerror(serrno));
       return;
      }

// Add the symlink
//
   addLink(Rfn, Lfn);
}
 
/******************************************************************************/
/*                                u n P r e p                                 */
/******************************************************************************/

void XrdCS2DCM::unPrep(const char *Fid)
{
   char *servP, *endP, Lfn[CA_MAXPATHLEN+1];
   unsigned long long Fnum;

// Convert the Fid to the Castor file name (lfn)
//
   if ((servP = index(Fid, '@')))
      {servP++;
       Fnum = strtoll(Fid, &endP, 10);
       if (*endP != '@') servP = 0;
      }
   if (!servP) 
      {XrdLog.Emsg("unPrep", "Invalid ffileid -", Fid);
       return;
      }

// Convert the fid to the actual Castor file name
//
   if (Cns_getpath(servP, Fnum, Lfn) != 0)
      {XrdLog.Emsg("unPrep", Fid, "fid conversion failed;", sstrerror(serrno));
       return;
      }

// Remove the symlink
//
   delLink(Lfn);
}
