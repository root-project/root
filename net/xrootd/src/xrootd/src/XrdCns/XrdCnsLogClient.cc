/******************************************************************************/
/*                                                                            */
/*                    X r d C n s L o g C l i e n t . c c                     */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCnsLogClientCVSID = "$Id$";
  
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include "Xrd/XrdTrace.hh"

#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientAdmin.hh"

#include "XrdCns/XrdCnsConfig.hh"
#include "XrdCns/XrdCnsInventory.hh"
#include "XrdCns/XrdCnsLog.hh"
#include "XrdCns/XrdCnsLogClient.hh"
#include "XrdCns/XrdCnsLogFile.hh"
#include "XrdCns/XrdCnsLogRec.hh"
#include "XrdCns/XrdCnsXref.hh"

#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucNSWalk.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysTimer.hh"

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/
  
namespace XrdCns
{
extern XrdCnsConfig Config;

extern XrdSysError  MLog;

extern XrdOucTrace  XrdTrace;
}

using namespace XrdCns;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCnsLogClient::XrdCnsLogClient(XrdOucTList     *rP,
                                 XrdCnsLogClient *pClient) : lfSem(0)
{
   static int cNum = 0;
   static int bSfx = static_cast<int>(time(0)) - 1248126834;
   static char *myName = XrdNetDNS::getHostName();
   char destBuff[512];

// Save our index into the commit array
//
   pfxNF    = cNum++;
   Next     = pClient;
   sfxFN    = bSfx;
   logFirst = 0;
   logLast  = 0;
   urlHost  = strdup(rP->text);

// Construct our logfile path (gauranteed to end with a slash)
//
   strcpy(logDir, Config.ePath);
   logFN = logDir + strlen(Config.ePath);
   strcpy(logFN, rP->text);
   logFN = logFN + strlen(rP->text);
   *logFN++ = '/';

// Estabish the file creation url
//
   crtFN = crtURL + sprintf(crtURL, "root://%s/", urlHost);

// Estalish the admin URL
//
   sprintf(destBuff, "root://%s//tmp", urlHost);
   admURL = strdup(destBuff);
   Admin = 0;

// Establish the backup operation processing
//
   arkOnly = Config.Opts & XrdCnsConfig::optNoCns;
   if (rP->val >= 0) {*arkURL = '\0'; arkFN = 0;}
      else {strcpy(arkURL, crtURL); arkPath = arkURL + strlen(crtURL);
            strcpy(arkPath,Config.bPath); strcat(arkPath, myName);
            arkFN  = arkPath + strlen(arkPath); *arkFN++ = '/';
            if (!arkOnly) arkOnly= (rP == Config.bDest);
            MLog.Emsg("LogClient", "Server inventory at", arkURL);
           }
}

/******************************************************************************/
/*                              A c t i v a t e                               */
/******************************************************************************/
  
int XrdCnsLogClient::Activate(XrdCnsLogFile *basefile)
{
   XrdCnsLogFile *lfP;

// Construct the our name for the file
//
   sfxFN++;
   sprintf(logFN,"cns.log.%d.%010d", pfxNF, sfxFN);

// Create new log file and subscribe it to the base file
//
   if ((lfP = basefile->Subscribe(logDir, pfxNF)))
      {lfMutex.Lock();
       if (logLast) logLast->Next = lfP;
          else logFirst = lfP;
       logLast = lfP; lfSem.Post();
       lfMutex.UnLock();
      }

// All done
//
   if (Next) return Next->Activate(basefile);
   return 1;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdCnsLogClient::Init()
{
   static const int Mode = S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH;
   XrdOucNSWalk::NSEnt *nInv = 0, *nFirst, *nsP;
   XrdCnsLogFile *fP;
   long long totsz = 0;
   int rc;

// Delete any partial inventory log file
//
   strcpy(logFN, XrdCnsLog::invFNa);
   unlink(logDir);

// Create a path if it does not exist
//
   if ((rc = XrdOucUtils::makePath(logDir, Mode)))
      {MLog.Emsg("Init", rc, "create log path", logDir); return 0;}

// Now get all of the log files in the directory
//
   *logFN = '\0';
   nFirst = XrdCnsLog::List(logDir, &nInv);

// If this is a recreate then only one log file will be processed
//
   if (Config.Opts & XrdCnsConfig::optRecr)
      while((nsP = nFirst))
           {nFirst = nFirst->Next; delete nsP;}

// If there is an inventory log, make sure it's first in the process list
//
   if (nInv) {nInv->Next = nFirst; nFirst = nInv;
              nInv->Stat.st_nlink = 0;
             }

// Document what we have while we create a log file list
//
   while((nsP = nFirst))
        {nFirst = nFirst->Next;
         MLog.Emsg("Init", "Recovered log file", nsP->Path);
         fP = new XrdCnsLogFile(nsP->Path, nsP->Stat.st_nlink, 0);
         if (logLast) logLast->Next = fP;
            else      logFirst      = fP;
         totsz += nsP->Stat.st_size;
         logLast = fP;
         delete nsP;
        }

// Now indicate if something is amiss
//
   if (totsz > 10*1024*1024)
      MLog.Emsg("Init", "Warning! More than 10MB of logs queued for", urlHost);

// If this is a create, then run the log file, otherwise return for a start
//
   if (!(Config.Opts & XrdCnsConfig::optRecr)) return 1;
   if (!(rc = Run(0)))
      MLog.Emsg("LogClient", urlHost, "namespace recreation failed!");
   return rc;
}
  
/******************************************************************************/
/*                                   R u n                                    */
/******************************************************************************/

int XrdCnsLogClient::Run(int Always)
{
   const char *TraceID = "ClientRun";
   XrdCnsLogFile *lfP = 0;
   XrdCnsLogRec  *lrP;
   char invDir[MAXPATHLEN+1], *invFN = invDir;
   time_t mCheck = time(0) - 10;
   int n, Ok = 0;

// This may be a one time excution to recreate the name space (with out
// without an inventory). Check if this is the case.
//
   if (!Always && !arkFN && !Manifest()) return 0;

// Process requests as they come in. We are always assured that we have at
// least one log file in the chain of log files. Note that log records
// returned by CnsLogFile are *not* recycleable!
//
   Admin = admConnect(Admin);

do{if (arkFN && time(0) >= mCheck)
      {if (!Manifest())
          {if (!Always) return 0;
           MLog.Emsg("LogClient","Unable to create inventory at",arkURL);
          }
       mCheck = time(0) + Config.mInt;
      }

   do {lfMutex.Lock();
       if ((lfP = logFirst))
          {if (!(logFirst = lfP->Next)) logLast = 0; lfMutex.UnLock();}
          else {lfMutex.UnLock(); lfSem.Wait();}
      } while(!lfP);

   if (lfP->Open())
      {while((lrP = lfP->getRec()))
            {if (arkOnly) continue;
             TRACE(DEBUG, urlHost <<" log data: '" <<lrP->Data() <<"'");
             switch (lrP->Type())
                    {case XrdCnsLogRec::lrClosew: Ok = do_Trunc (lrP); break;
                     case XrdCnsLogRec::lrCreate: Ok = do_Create(lrP); break;
                     case XrdCnsLogRec::lrInvD:   strcpy(invDir, lrP->Lfn1(n));
                                                  invFN = invDir+n; Ok = 0;
                                                  *invFN++ = '/';
                                                  break;
                     case XrdCnsLogRec::lrInvF:   strcpy(invFN, lrP->Lfn1());
                                        if ((Ok = do_Create(lrP, invDir)))
                                             Ok = do_Trunc( lrP, invDir);
                                                  break;
                     case XrdCnsLogRec::lrMkdir:  Ok = do_Mkdir (lrP); break;
                     case XrdCnsLogRec::lrMv:     Ok = do_Mv    (lrP); break;
                     case XrdCnsLogRec::lrRm:     Ok = do_Rm    (lrP); break;
                     case XrdCnsLogRec::lrRmdir:  Ok = do_Rmdir (lrP); break;
                     case XrdCnsLogRec::lrMount:
                     case XrdCnsLogRec::lrSpace:
                          if (Config.Space)
                             Config.Space->Add(lrP->Lfn1(),lrP->Space());
                                                                       break;
                     case XrdCnsLogRec::lrTOD:                         break;
                     default: MLog.Emsg("Run","Invalid logrec for",lrP->Lfn1());
                              Ok = 0;
                    }
             if (Ok) lfP->Commit();
            }
       if (!arkFN || Archive(lfP)) lfP->Unlink();
       delete lfP;
      }
  } while(Always);

// We get here only for 1-time command processing (unthreaded)
//
   return 1;
}
  
/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
namespace XrdCns
{
void *StartLogClient(void *parg)
{
   XrdCnsLogClient *lcP = static_cast<XrdCnsLogClient *>(parg);
   lcP->Run();
   return (void *)0;
}
}

int XrdCnsLogClient::Start()
{
   pthread_t tid;
   int rc;

// Start the log client
//
   if ((rc = XrdSysThread::Run(&tid, StartLogClient, (void *)this,
                                 XRDSYSTHREAD_BIND, "Log client")))
      {MLog.Emsg("Start", rc, "create log client thread");
       if (Next) Next->Start();
       return 0;
      }

// All done
//
   if (Next) return Next->Start();
   return 1;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                            a d m C o n n e c t                             */
/******************************************************************************/

XrdClientAdmin *XrdCnsLogClient::admConnect(XrdClientAdmin *adminP)
{
   const char *TraceID = "admConnect";
   static XrdSysMutex xcMutex;

// If we have a previous instance of the admin, delete it
//
   if (adminP) delete adminP;

// Get a new admin
//
   xcMutex.Lock();
   adminP = new XrdClientAdmin(admURL);
   xcMutex.UnLock();

// Loop until connected
//
   do {TRACE(DEBUG, "Connecting to " <<urlHost);
       if (adminP->Connect()) break;
       xrdEmsg("connect", admURL, adminP);
       XrdSysTimer::Snooze(20);
      } while (1);

// All done
//
   return adminP;
}

/******************************************************************************/
/*                               A r c h i v e                                */
/******************************************************************************/
  
int XrdCnsLogClient::Archive(XrdCnsLogFile *lfP)
{
   static const int OMode = kXR_open_updt | kXR_delete | kXR_mkpath;
   static const int AMode = kXR_ur | kXR_uw | kXR_gr | kXR_gw | kXR_or;
   XrdClient *fP;
   int   Blen, rc = 1;
   const char *lFN;
   char *oP, oldName[2048], *Buff = lfP->getLog(Blen);

// See if there is anything to archive
//
   if (!arkFN || !Blen) return 1;

// Get the log file name (we can't use our object's one)
//
   if (!(lFN = rindex(lfP->FName(), '/')))
      {MLog.Emsg("LogClient", "Unable to determine archive log file name.");
       return 0;
      } else lFN++;

// Get a client instance
//
   strcpy(arkFN, lFN);
   MLog.Emsg("Archive", "Creating backup", arkURL);
   *arkFN = '.';
   fP = new XrdClient(arkURL);

// Open the target file and write out the log and close the file
//
   if (!fP->Open(AMode, OMode, 0) || (fP->LastServerResp()->status) != kXR_ok)
      xrdEmsg("archive", lfP->FName(), fP);
      else if (Buff && Blen && !fP->Write(Buff, 0, Blen))
              xrdEmsg("write", lfP->FName(), fP);
              else rc = 0;

// Rename the file to what it really should be
//
   delete fP;
   strcpy(oldName, arkURL); *arkFN = (*lFN == 'i' ? 'I' : *lFN);
   oP = oldName + (arkPath - arkURL);
   if (!Admin->Mv(oP, arkPath))
      {xrdEmsg("rename", oldName, Admin); Admin->Rm(oP);  rc = 1;}
   
   return rc == 0;
}
  
/******************************************************************************/
/*                             d o _ C r e a t e                              */
/******************************************************************************/
  
int XrdCnsLogClient::do_Create(XrdCnsLogRec *lrP, const char *lfn)
{
   static const int OMode = kXR_open_updt | kXR_delete | kXR_mkpath;
   XrdClient *fP;
   int AMode = kXR_ur | kXR_uw;
   int CMode, Ok = 1;

// Construct the Mode
//
   CMode = lrP->Mode();
   if (CMode & S_IRGRP) AMode |= kXR_gr;
   if (CMode & S_IWGRP) AMode |= kXR_gw;
   if (CMode & S_IROTH) AMode |= kXR_or;

// Get a client instance
//
   if (!lfn) strcpy(crtFN, lrP->Lfn1());
      else {strcpy(crtFN, lfn);
            if (Config.Space)
               {char *spName = Config.Space->Key(lrP->Space());
                if (spName && strcmp(spName, "public"))
                   {strcat(crtFN, "?oss.cgroup="); strcat(crtFN, spName);}
               }
           }
   fP = new XrdClient(crtURL);

// Open the target file and write out the log
//
   if (!fP->Open(AMode, OMode, 0) || (fP->LastServerResp()->status) != kXR_ok)
      Ok = xrdEmsg("create", lrP->Lfn1(), fP);

// Finish up
//
   delete fP;
   return Ok;
}
  
/******************************************************************************/
/*                              d o _ M k d i r                               */
/******************************************************************************/
  
int XrdCnsLogClient::do_Mkdir(XrdCnsLogRec *lrP)
{
   if (!Admin->Mkdir(lrP->Lfn1(), 7, 7, 5))
      return xrdEmsg("mkdir", lrP->Lfn1());
   return 1;
}

/******************************************************************************/
/*                                 d o _ M v                                  */
/******************************************************************************/
  
int XrdCnsLogClient::do_Mv(XrdCnsLogRec *lrP)
{
   if (!Admin->Mv(lrP->Lfn1(), lrP->Lfn2()))
      return xrdEmsg("mv", lrP->Lfn1());
   return 1;
}

/******************************************************************************/
/*                                 d o _ R m                                  */
/******************************************************************************/
  
int XrdCnsLogClient::do_Rm(XrdCnsLogRec *lrP)
{
   if (!Admin->Rm(lrP->Lfn1())) return xrdEmsg("rm", lrP->Lfn1());
   return 1;
}

/******************************************************************************/
/*                              d o _ R m d i r                               */
/******************************************************************************/
  
int XrdCnsLogClient::do_Rmdir(XrdCnsLogRec *lrP)
{
   if (!Admin->Rmdir(lrP->Lfn1())) return xrdEmsg("rmdir", lrP->Lfn1());
   return 1;
}

/******************************************************************************/
/*                              d o _ T r u n c                               */
/******************************************************************************/
  
int XrdCnsLogClient::do_Trunc(XrdCnsLogRec *lrP, const char *lfn)
{
   if (!Admin->Truncate((lfn ? lfn : lrP->Lfn1()), lrP->Size()))
      return xrdEmsg("trunc", (lfn ? lfn : lrP->Lfn1()));
   return 1;
}

/******************************************************************************/
/*                              M a n i f e s t                               */
/******************************************************************************/

int XrdCnsLogClient::Manifest()
{
   const char *TraceID = "Manifest";
   XrdCnsInventory Inventory;
   
   XrdCnsLogFile *lfP;
   XrdOucTList *xP;
   long long vSize;
   long V1, V2, V3;
   char oldName[MAXPATHLEN+1];

// Check if we will be processing an inventory log file
//
   lfMutex.Lock();
   if (logFirst)
      {const char *fN = rindex(logFirst->FName(), '/');
       if (fN && !strcmp(XrdCnsLog::invFNz, fN+1))
          {lfMutex.UnLock(); return 1;}
      }
   lfMutex.UnLock();

// Check if we have an inventory file at the destination
//
   if (arkFN)
      {strcpy(arkFN, XrdCnsLog::invFNz);
       if (Admin->Stat(arkPath, V1, vSize, V2, V3)) return 1;
       if (Admin->LastServerError()->errnum != kXR_NotFound)
          {xrdEmsg("find inventory", arkPath, Admin); return 0;}
       TRACE(DEBUG, "Creating inventory...");
      }

// Create a log file for the inventory
//
   strcpy(logFN, XrdCnsLog::invFNa);
   lfP = new XrdCnsLogFile(logDir, 0, 0);
   if (!(lfP->Open(0))) {delete lfP; return 0;}

// Initialize inventory processing
//
   Inventory.Init(lfP);

// Now inventory all the exported paths
//
   xP = Config.Exports;
   while(xP && Inventory.Conduct(xP->text)) xP = xP->next;
   lfP->Eol(); delete lfP;

// Check if all went well
//
   if (xP) {unlink(logDir); return 0;}

// Rename the file to what we really want it to be
//
   strcpy(oldName, logDir); strcpy(logFN, XrdCnsLog::invFNt);
   if (rename(oldName, logDir))
      {MLog.Emsg("Manifest", errno, "rename", oldName);
       unlink(logDir); return 0;
      }

// Create a new log file object to handle this log file and chain it in
//
   lfP = new XrdCnsLogFile(logDir, 0, 0);
   lfMutex.Lock();
   lfP->Next = logFirst; logFirst = lfP;
   if (!logLast) logLast = lfP;
   lfMutex.UnLock();

// All done
//
   return 1;
}
 
/******************************************************************************/
/*                              m a p E r r o r                               */
/******************************************************************************/
  
int XrdCnsLogClient::mapError(int rc)
{
    switch(rc)
       {case kXR_NotFound:      return ENOENT;
        case kXR_NotAuthorized: return EACCES;
        case kXR_IOError:       return EIO;
        case kXR_NoMemory:      return ENOMEM;
        case kXR_NoSpace:       return ENOSPC;
        case kXR_ArgTooLong:    return ENAMETOOLONG;
        case kXR_noserver:      return EHOSTUNREACH;
        case kXR_NotFile:       return ENOTBLK;
        case kXR_isDirectory:   return EISDIR;
        case kXR_FSError:       return ENOSYS;
        default:                return ECANCELED;
       }
}
  
/******************************************************************************/
/*                               x r d E m s g                                */
/******************************************************************************/

int XrdCnsLogClient::xrdEmsg(const char *Opname, const char *theFN,
                             XrdClientAdmin *aP)
{
   char *etext  =  aP->LastServerError()->errmsg;
   int rc=mapError(aP->LastServerError()->errnum);

   if (rc == ECANCELED && etext && *etext) MLog.Emsg("LogClient", etext);
      else MLog.Emsg("LogClient", rc, Opname, theFN);
   return 0;
}

/******************************************************************************/

int XrdCnsLogClient::xrdEmsg(const char *Opname, const char *theFN)
{
   return xrdEmsg(Opname, theFN, Admin);
}

/******************************************************************************/

int XrdCnsLogClient::xrdEmsg(const char *Opn, const char *Fn, XrdClient *fP)
{
   char *etext  =  fP->LastServerError()->errmsg;
   int rc=mapError(fP->LastServerError()->errnum);

   if (rc == ECANCELED && etext && *etext) MLog.Emsg("LogClient", etext);
      else MLog.Emsg("LogClient", rc, Opn, Fn);
   return 0;
}
