/******************************************************************************/
/*                                                                            */
/*                          X r d F r m C I D . c c                           */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdFrmCIDCVSID = "$Id$";

#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>

#include "XrdFrm/XrdFrmCID.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/
  
XrdFrmCID   XrdFrm::CID;

XrdSysMutex XrdFrmCID::cidMon::cidMutex;
  
/******************************************************************************/
/*                                   A d d                                    */
/******************************************************************************/
  
int XrdFrmCID::Add(const char *iName, const char *cName, time_t addT, pid_t Pid)
{
   cidMon cidMonitor;
   cidEnt *cP;
   int ckp = 0;

// If this is a new entry, create one
//
   if (!(cP = Find(iName)))
      {First = new cidEnt(First, iName, cName, addT, Pid);
       if (!strcmp(iName, "anon")) Dflt = First;
       Update();
       return 1;
      }

// Ignore this update if this request is older than the previous one
//
   if (cP->addT >= addT) return 0;

// Update existing entry
//
   if (strcmp(cP->cName, cName))
      {free(cP->cName);
       cP->cName = strdup(cName);
       cP->cNLen = strlen(cName);
       ckp = 1;
      }
   if (cP->Pid != Pid) {cP->Pid = Pid; ckp = 1;}
   cP->addT = addT;
   if (ckp) Update();
   return ckp;
}
  
/******************************************************************************/
/* Private:                         F i n d                                   */
/******************************************************************************/

XrdFrmCID::cidEnt *XrdFrmCID::Find(const char *iName)
{
   cidEnt *cP;

// If no instance name, then return default
//
   if (!iName || !(*iName)) return Dflt;


// Prepare to find the name
//
   cP = First;
   while(cP && strcmp(iName, cP->iName)) cP = cP->Next;

// Return result
//
   return cP;
}

/******************************************************************************/
/* Public:                           G e t                                    */
/******************************************************************************/
  
int XrdFrmCID::Get(const char *iName, char *buff, int blen)
{
   cidMon cidMonitor;
   cidEnt *cP;

// Find the entry
//
   if (!(cP = Find(iName))) {*buff = 0; return 0;}

// Copy out the cluster name
//
   strlcpy(buff, cP->cName, blen);
   return 1;
}
/******************************************************************************/
  
int XrdFrmCID::Get(const char *iName, const char *vName, XrdOucEnv *evP)
{
   cidMon cidMonitor;
   cidEnt *cP;

// Find the entry
//
   if (!(cP = Find(iName))) return 0;

// Set cluster name in the environment
//
   if (vName && evP) evP->Put(vName, cP->cName);
   return 1;
}
  
/******************************************************************************/
/* Public:                          I n i t                                   */
/******************************************************************************/

int XrdFrmCID::Init(const char *aPath)
{
   EPNAME("Init");
   XrdOucStream cidFile(&Say);
   char Path[1024], *lP, *Pfn;
   int cidFD, n, NoGo = 0;

// Construct the appropriate file names
//
   strcpy(Path, aPath);
   n = strlen(aPath);
   if (Path[n-1] != '/') Path[n++] = '/';
   Pfn = Path+n;
   strcpy(Pfn, "CIDS.new"); cidFN2 = strdup(Path);
   strcpy(Pfn, "CIDS");     cidFN  = strdup(Path);

// Try to open the cluster checkpoint file.
//
   if ( (cidFD = open(cidFN, O_RDONLY, 0)) < 0)
      {if (errno == ENOENT) return 0;
       Say.Emsg("Init", errno, "open cluster chkpnt file", cidFN);
       return 1;
      }
   cidFile.Attach(cidFD);

// Now start reading records until eof.
//
   while((lP = cidFile.GetLine()))
        if (*lP)
           {DEBUG("Recovering cid entry: " <<lP);
            NoGo |= Init(cidFile);
           }

// Now check if any errors occured during file i/o
//
   if (NoGo) Say.Emsg("Init", "Errors processing chkpnt file", cidFN);
      else if ((n = cidFile.LastError()))
              NoGo = Say.Emsg("Init", n, "read cluster chkpnt file", cidFN);
   cidFile.Close();

// Return final return code
//
   return NoGo;
}

/******************************************************************************/

int XrdFrmCID::Init(XrdOucStream &cidFile)
{
   EPNAME("Init");
   char *iP, *cP, *tP, *uP;
   time_t addT;
   pid_t  Pid;

// The record is <iname> <cname> <addt> <pid>
//
   if (!(iP = cidFile.GetToken()))
      {Say.Emsg("Init","Missing cluster instance name."); return 1;}
   if (!(cP = cidFile.GetToken()))
      {Say.Emsg("Init","Missing cluster name for", iP);   return 1;}
   if (!(tP = cidFile.GetToken()))
      {Say.Emsg("Init","Missing timestamp for", iP);      return 1;}
   addT = static_cast<time_t>(strtoll(tP, &uP, 10));
   if (!addT || *uP)
      {Say.Emsg("Init","Invalid timestamp for", iP);      return 1;}
   if (!(tP = cidFile.GetToken()))
      {Say.Emsg("Init","Missing process id for",   iP);   return 1;}
   Pid  = static_cast<pid_t>(strtol(tP, &uP, 10));
   if (*uP)
      {Say.Emsg("Init","Invalid process id for", iP);     return 1;}

// Validate the process ID
//
   if (Pid && kill(Pid, 0) < 0 && errno == ESRCH)
      {DEBUG("Process " <<Pid <<" not found for instance " <<iP);
       Pid = 0;
      }

// Now add the entry
//
   First = new cidEnt(First, iP, cP, addT, Pid);
   if (!strcmp(iP, "anon")) Dflt = First;
   return 0;
}

/******************************************************************************/
/* Public:                           R e f                                    */
/******************************************************************************/
  
void XrdFrmCID::Ref(const char *iName)
{
   cidMon cidMonitor;
   cidEnt *cP;

// Find the entry
//
   if ((cP = Find(iName))) cP->useCnt = 1;
}

/******************************************************************************/
/* Private:                       U p d a t e                                 */
/******************************************************************************/
  
int XrdFrmCID::Update()
{
   EPNAME("Update");
   static char buff[40];
   static const int Mode = S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH;
   static struct iovec iov[] = {{0,0}, {(char *)" ", 1}, // 0: Instance
                                {0,0},                   // 2: Cluster
                                {buff, 0}};              // 3: Timestamp pid
   static const int iovn = sizeof(iov)/sizeof(struct iovec);
   FLOCK_t lock_args;
   cidEnt *cP = First, *cPP = 0, *cPN;
   int rc, cidFD;

// Open the temp file first in r/w mode
//
   if ((cidFD = open(cidFN2, O_RDWR|O_CREAT, Mode)) < 0)
      {Say.Emsg("Init",errno,"open",cidFN2);
       return 0;
      }
   fcntl(cidFD, F_SETFD, FD_CLOEXEC);

// Lock the file
//
   bzero(&lock_args, sizeof(lock_args));
   lock_args.l_type = F_WRLCK;
   do {rc = fcntl(cidFD,F_SETLKW,&lock_args);} while(rc < 0 && errno == EINTR);
   if (rc < 0)
      {Say.Emsg("Update", errno, "lock", cidFN2);
       close(cidFD);
       return 0;
      }

// Now truncate the file to zero
//
   if (ftruncate(cidFD, 0) < 0)
      {Say.Emsg("Update", errno, "truncate", cidFN2);
       close(cidFD);
       return 0;
      }

// Write out the cluster information
//
   while(cP)
        {if (!(cP->Pid) && !(cP->useCnt) && strcmp(cP->iName, "anon"))
            {DEBUG("Removing dead instance " <<cP->iName);
             if (cPP) cPN = cPP->Next = cP->Next;
                else  cPN = First     = cP->Next;
             delete cP;
             cP = cPN;
             continue;
            }
         iov[0].iov_base = cP->iName; iov[0].iov_len = cP->iNLen;
         iov[2].iov_base = cP->cName; iov[2].iov_len = cP->cNLen;
         iov[3].iov_len = sprintf(buff, " %ld %d",
                          static_cast<long>(cP->addT),
                          static_cast<int> (cP->Pid)) + 1;
         if (writev(cidFD, iov, iovn) < 0)
            {Say.Emsg("Update", errno, "writing", cidFN2);
             close(cidFD);
             return 0;
            }
         cPP = cP; cP = cP->Next;
        }

// Now rename the file to be the original while we hav the file open
//
   if (rename(cidFN2, cidFN) < 0)
      {Say.Emsg("Update", errno, "rename", cidFN2);
       close(cidFD);
       return 0;
      }

// All done
//
   close(cidFD);
   return 1;
}
