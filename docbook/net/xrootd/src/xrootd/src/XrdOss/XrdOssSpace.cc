/******************************************************************************/
/*                                                                            */
/*                        X r d O s s S p a c e . c c                         */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$
 
const char *XrdOssSpaceCVSID = "$Id$";

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>

#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                   G l o b a l s   a n d   S t a t i c s                    */
/******************************************************************************/

extern XrdSysError OssEroute;

       const char       *XrdOssSpace::qFname                     = 0;
       const char       *XrdOssSpace::uFname                     = 0;
       XrdOssSpace::uEnt XrdOssSpace::uData[XrdOssSpace::maxEnt] = {{{0}}};
       short             XrdOssSpace::uDvec[XrdOssSpace::maxEnt] = {0};
       int               XrdOssSpace::fencEnt                    = 0;
       int               XrdOssSpace::freeEnt                    =-1;
       int               XrdOssSpace::aFD                        =-1;
       int               XrdOssSpace::Solitary                   = 0;
       time_t            XrdOssSpace::lastMtime                  = 0;

/******************************************************************************/
/*                                A d j u s t                                 */
/******************************************************************************/
  
void XrdOssSpace::Adjust(int Gent, off_t Space, sType stNum)
{
   int offset, unlk = 0;
   int uOff = offsetof(uEnt,Bytes[0]) + (sizeof(long long)*stNum);

// Verify the entry number
//
   if (Gent < 0 || Gent >= fencEnt) return;
   offset = sizeof(uEnt)*Gent + uOff;

// For stand-alone processes, we need to convert server adjustments to make
// the update inter-process safe.
//
   if (Solitary && stNum == Serv) stNum = (Space > 0 ? Pstg : Purg);

// Check if we need a lock and a refresh. For admin stats we need to make the
// result idempotent w.r.t. updates by convoluting pstg/purg space numbers.
//
   if (stNum != Serv)
      {if (!UsageLock()) return;
       if (pread(aFD, &uData[Gent], sizeof(uEnt), offset-uOff) < 0)
          {OssEroute.Emsg("Adjust", errno, "read usage file", uFname);
           UsageLock(0); return;
          }
       if (stNum == Admin)
          {uData[Gent].Bytes[Admin] = 0;
           Space = Space - uData[Gent].Bytes[Pstg] + uData[Gent].Bytes[Purg];
          }
       unlk = 1;
      }

// Update the space statistic (protected by caller's mutex)
//
   if ((uData[Gent].Bytes[stNum] += Space) < 0 && stNum != Admin)
      uData[Gent].Bytes[stNum] = 0;

// Write out the the changed field. For servers, we can do this without a lock
// because we are the only ones allowed to write this field.
//
   if (pwrite(aFD, &uData[Gent].Bytes[stNum], ULen, offset) < 0)
      OssEroute.Emsg("Adjust", errno, "update usage file", uFname);

// Unlock the file if we locked it
//
   if (unlk) UsageLock(0);
}

/******************************************************************************/
  
void XrdOssSpace::Adjust(const char *GName, off_t Space, sType stNum)
{
   int i;

// Try to find the current entry in the file
//
   if ((i = findEnt(GName)) >= 0) Adjust(i, Space, stNum);
}

/******************************************************************************/
/*                                A s s i g n                                 */
/******************************************************************************/
  
int XrdOssSpace::Assign(const char *GName, long long &Usage)
{
   off_t offset;
   int i;

// Try to find the current entry in the file
//
   if ((i = findEnt(GName)) >= 0)
      {Usage = uData[i].Bytes[Serv];
       return i;
      }

// See if we can create a new entry
//
   Usage = 0;
   if (freeEnt >= maxEnt || freeEnt < 0)
      {OssEroute.Emsg("Assign", uFname, "overflowed for", GName);
       return -1;
      }

// Create the entry
//
   if (!UsageLock()) return -1;
   memset(&uData[freeEnt], 0, sizeof(uEnt));
   strcpy(uData[freeEnt].gName, GName);
   uData[freeEnt].Bytes[addT] = static_cast<long long>(time(0));
   offset = sizeof(uEnt) * freeEnt;
   if (pwrite(aFD, &uData[freeEnt], sizeof(uEnt), offset) < 0)
      {OssEroute.Emsg("Adjust", errno, "update usage file", uFname);
       UsageLock(0); return -1;
      }
   UsageLock(0);

// Add this to the vector table
//
   uDvec[fencEnt++] = i = freeEnt;

// Find next free entry
//
   for (freeEnt = freeEnt+1; freeEnt < maxEnt; freeEnt++)
       if (*uData[freeEnt].gName == '\0') break;

// All done here
//
   return i;
}

/******************************************************************************/
/*                               f i n d E n t                                */
/******************************************************************************/
  
int XrdOssSpace::findEnt(const char *GName)
{
   int i;

// Try to find the current entry in the file
//
   for (i = 0; i < fencEnt; i++)
       if (!strcmp(uData[uDvec[i]].gName, GName)) return i;
   return -1;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdOssSpace::Init() {return (uFname ? haveUsage:0) | (qFname ? haveQuota:0);}

/******************************************************************************/
  
int XrdOssSpace::Init(const char *aPath, const char *qPath, int isSOL)
{
   static const mode_t theMode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP;
   struct stat buf;
   const char *iP;
   char *aP, buff[1048];
   int i, opts, updt = 0;

// Indicate whether we are solitary or not
//
   Solitary = isSOL;

// Handle quota file first
//
   if (qPath)
      {qFname = strdup(qPath);
       if (!Quotas()) return 0;
       XrdOucEnv::Export("XRDOSSQUOTAFILE", qFname);
      }

// Construct the file path for the usage file
//
   if (!aPath) return 1;
   strcpy(buff, aPath);
   aP = buff + strlen(aPath);
   if (*(aP-1) != '/') *aP++ = '/';
   if ((iP = XrdOucUtils::InstName(-1)))
      {strcpy(aP, iP); aP += strlen(iP); *aP++ = '/'; *aP = '\0';
       mkdir(buff, S_IRWXU | S_IRWXG);
      }
   strcpy(aP, ".Usage");
   uFname = strdup(buff);
   XrdOucEnv::Export("XRDOSSUSAGEFILE", uFname);

// First check if the file really exists, if not, create it
//
   if (stat(uFname, &buf))
      if (errno != ENOENT)
         {OssEroute.Emsg("Init", errno, "open", uFname);
          return 0;
         } else opts = O_CREAT|O_TRUNC;
      else if ( buf.st_size != DataSz && buf.st_size)
              {OssEroute.Emsg("Init", uFname, "has invalid size."); return 0;}
              else opts = 0;

// Open the target file
//
   if ((aFD = open(uFname, opts|O_RDWR|O_SYNC, theMode)) < 0)
      {OssEroute.Emsg("Init", errno, "open", uFname);
       return 0;
      }

// Lock the file
//
   UsageLock();

// Either read the contents or initialize the contents
//
   if (opts & O_CREAT || buf.st_size == 0)
      {memset(uData, 0, sizeof(uData));
       if (!write(aFD, uData, sizeof(uData)))
          {OssEroute.Emsg("Init", errno, "create", uFname);
           UsageLock(0); return 0;
          }
       fencEnt = 0; freeEnt = 0;
     } else {
      if (!read(aFD, uData, sizeof(uData)))
         {OssEroute.Emsg("Init", errno, "read", uFname);
          UsageLock(0); return 0;
         }
      for (i = 0; i < maxEnt; i++)
          {if (*uData[i].gName != '\0')
              {uDvec[fencEnt++] = i; updt = Readjust(i);}
              else if (freeEnt < 0) freeEnt = i;
          }
      if (freeEnt < 0) OssEroute.Emsg("Init", uFname, "is full.");
     }

// If we need to rewrite the data, do so
//
   if (updt && pwrite(aFD, uData, sizeof(uData), 0) < 0)
      OssEroute.Emsg("Init", errno, "rewrite", uFname);

// All done
//
   UsageLock(0); 
   sprintf(buff, "%d usage log entries in use; %d available.", 
                 fencEnt, maxEnt-fencEnt);
   OssEroute.Emsg("Init", buff);
   return 1;
}

/******************************************************************************/
/*                                Q u o t a s                                 */
/******************************************************************************/
  
int XrdOssSpace::Quotas()
{
  XrdOucStream Config(&OssEroute);
  XrdOssCache_Group *fsg;
  struct stat buf;
  long long qval;
  char cgroup[minSNbsz], *val;
  int qFD, NoGo = 0;

// See if the file has changed (note the firs time through it will have)
//
   if (stat(qFname,&buf))
      {OssEroute.Emsg("Quotas", errno, "process quota file", qFname);
       return 0;
      }
   if (buf.st_mtime == lastMtime) return 0;
   lastMtime = buf.st_mtime;

// Try to open the quota file.
//
   if ( (qFD = open(qFname, O_RDONLY, 0)) < 0)
      {OssEroute.Emsg("Quotas", errno, "open quota file", qFname);
       return 0;
      }

// Attach the file to a stream and tell people what we are doing
//
   OssEroute.Emsg("Quotas", "Processing quota file", qFname);
   Config.Attach(qFD);

// Now start reading records until eof.
//
   while((val = Config.GetMyFirstWord()))
        {if (strlen(val) >= sizeof(cgroup))
            {OssEroute.Emsg("Quotas", "invalid quota group =", val);
             NoGo = 1; continue;
            }
         strcpy(cgroup, val);

         if (!(val = Config.GetWord()))
            {OssEroute.Emsg("Quotas", "quota value not specified for", cgroup);
             NoGo = 1; continue;
            }
         if (XrdOuca2x::a2sz(OssEroute, "quota", val, &qval))
            {NoGo = 1; continue;
            }
         fsg = XrdOssCache_Group::fsgroups;
         while(fsg && strcmp(cgroup, fsg->group)) fsg = fsg->next;
         if (fsg) fsg->Quota = qval;
         if (!strcmp("public", cgroup)) XrdOssCache_Group::PubQuota = qval;
            else if (!fsg) OssEroute.Emsg("Quotas", cgroup, 
                                     "cache group not found; quota ignored");
        }
    close(qFD);
    return (NoGo ? 0 : 1);
}

/******************************************************************************/
/*                              R e a d j u s t                               */
/******************************************************************************/
  
int XrdOssSpace::Readjust()
{
   static time_t lastUtime = 0;
   struct stat buf;
   int k, rwsz, updt = 0;

// No readjustment needed if we are not a server or we have nothing
//
   if (fencEnt <= 0) return 0;
   if (!fstat(aFD, &buf))
      {if (buf.st_mtime == lastUtime) return 0;
       lastUtime = buf.st_mtime;
      }
   rwsz = sizeof(uEnt)*(uDvec[fencEnt-1] + 1);

// Lock the file
//
   UsageLock();

// Read the file again
//
   if (!pread(aFD, uData, rwsz, 0))
      {OssEroute.Emsg("Readjust", errno, "read", uFname);
       UsageLock(0); return 0;
      }

// Perform necessary readjustments but only for things we know about
//
   for (k = 0; k < fencEnt; k++) updt |= Readjust(uDvec[k]);

// If we need to rewrite the data, do so
//
   if (updt && pwrite(aFD, uData, rwsz, 0) < 0)
      OssEroute.Emsg("Readjust", errno, "rewrite", uFname);

// All done
//
   UsageLock(0);
   return updt;
}

/******************************************************************************/
  
int XrdOssSpace::Readjust(int i)
{

// Check if any readjustment is needed
//
   if (uData[i].Bytes[Pstg] || uData[i].Bytes[Purg] || uData[i].Bytes[Admin])
      {uData[i].Bytes[Serv] = uData[i].Bytes[Serv] + uData[i].Bytes[Pstg]
                            - uData[i].Bytes[Purg] + uData[i].Bytes[Admin];
       uData[i].Bytes[Pstg] = uData[i].Bytes[Purg] = uData[i].Bytes[Admin] = 0;
       return 1;
      }
   return 0;
}

/******************************************************************************/
/*                              U n a s s i g n                               */
/******************************************************************************/
  
int XrdOssSpace::Unassign(const char *GName)
{
   off_t offset;
   int k, i;

// Try to find the current entry in the file
//
   for (k = 0; k < fencEnt; k++)
       if (!strcmp(uData[uDvec[k]].gName, GName)) break;
   if (k >= fencEnt) return -1;
   i = uDvec[k];

// Create the entry
//
   if (!UsageLock()) return -1;
   memset(&uData[i], 0, sizeof(uEnt));
   offset = sizeof(uEnt) * i;
   if (pwrite(aFD, &uData[freeEnt], sizeof(uEnt), offset) < 0)
      {OssEroute.Emsg("Unassign", errno, "update usage file", uFname);
       UsageLock(0); return -1;
      }
   UsageLock(0);

// Squish out the uDvec
//
   if (i < freeEnt) freeEnt = i;
   for (i = k+1; i < fencEnt; i++) uDvec[k++] = uDvec[i];
   fencEnt--;
   return 0;
}
  
/******************************************************************************/
/*                                 U s a g e                                  */
/******************************************************************************/
  
long long XrdOssSpace::Usage(const char *GName, struct uEnt &uVal, int rrd)
{
   int i, rwsz;

// If we need to re-read the file, do so
//
   if (rrd)
      {if (fencEnt <= 0) return -1;
       UsageLock();
       rwsz = sizeof(uEnt)*(uDvec[fencEnt-1] + 1);
       if (!pread(aFD, uData, rwsz, 0))
          {OssEroute.Emsg("Readjust", errno, "read", uFname);
           UsageLock(0); return -1;
          }
       UsageLock(0);
      }

// Try to find the current entry in the file
//
   if ((i = findEnt(GName)) >= 0)
      {uVal = uData[i];
       return uData[i].Bytes[Serv];
      }

// Not found
//
   memset(&uVal, 0, sizeof(uEnt));
   return -1;
}

/******************************************************************************/
/* private:                    U s a g e L o c k                              */
/******************************************************************************/
  
int XrdOssSpace::UsageLock(int Dolock)
{
   FLOCK_t lock_args;
   const char *What;
   int rc;

// Establish locking options
//
   bzero(&lock_args, sizeof(lock_args));
   if (Dolock) {lock_args.l_type = F_WRLCK; What =   "lock";}
      else     {lock_args.l_type = F_UNLCK; What = "unlock";}

// Perform action.
//
   do {rc = fcntl(aFD,F_SETLKW,&lock_args);} while(rc < 0 && errno == EINTR);
   if (rc < 0) {OssEroute.Emsg("UpdateLock", errno, What, uFname); return 0;}

// All done
//
   return 1;
}
