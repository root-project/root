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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>

#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysError.hh"

/******************************************************************************/
/*                   G l o b a l s   a n d   S t a t i c s                    */
/******************************************************************************/

extern XrdSysError OssEroute;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOssSpace::XrdOssSpace(const char *apath, const char *qfile)
{
   char *iP, *aP, buff[1024];

// Construct the file path for the usage file
//
   if (apath)
      {strcpy(buff, apath);
       aP = buff + strlen(apath);
       if (*(aP-1) != '/') *aP++ = '/';
       if ((iP = getenv("XRDNAME")) && *iP && strcmp(iP, "anon"))
          {strcpy(aP, iP); aP += strlen(iP); *aP++ = '/'; *aP = '\0';
           mkdir(buff, S_IRWXU | S_IRWXG);
          }
       strcpy(aP, "Usage");
       aFname = strdup(buff);
       nextEnt = 0;
      } else aFname = 0;

// Get file path for quota file
//
   QFile = (qfile ? strdup(qfile) : 0);
   lastMtime = 0;
}

/******************************************************************************/
/*                                A d j u s t                                 */
/******************************************************************************/
  
void XrdOssSpace::Adjust(int Gent, off_t Space)
{
   static const int uOff = offsetof(uEnt,Used);
   int offset;

// Verify the entry number
//
   if (Gent < 0 || Gent >= nextEnt) return;

// Update the space statistic (protected by caller's mutex)
//
   if ((uData[Gent].Used += Space) < 0) uData[Gent].Used = 0;

// Write out the the changed field
//
   offset = sizeof(uEnt)*Gent + uOff;
   if (pwrite(aFD, &uData[Gent].Used, sizeof(uData[0].Used), offset) < 0)
      OssEroute.Emsg("Adjust", errno, "update usage file", aFname);
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
   for (i = 0; i < nextEnt; i++)
       if (!strcmp(uData[i].gName, GName)) break;

// Check if we should create a new entry or return an existing one
//
   if (i >= nextEnt)
      {Usage = 0;
       if (i >= maxEnt)
         {OssEroute.Emsg("Assign", aFname, "overflow for", GName);
          return -1;
         } else {
          memset(&uData[i], 0, sizeof(uEnt));
          strcpy(uData[i].gName, GName); nextEnt++;
          offset = sizeof(uEnt) * i;
          if (pwrite(aFD, &uData[i], sizeof(uEnt), offset) < 0)
            {OssEroute.Emsg("Adjust", errno, "update usage file", aFname);
             return -1;
            }
         }
      } else Usage = uData[i].Used;

// All done here
//
   return i;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/
  
int XrdOssSpace::Init()
{
   static const mode_t theMode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP;
   struct stat buf;
   char buff[512];
   int i, opts;

// First handle the quotas and check if we have a usage file
//
   if (QFile && !Quotas()) return 0;
   if (!aFname) return 1;

// First check if the file really exists, if not, create it
//
   if (stat(aFname, &buf))
      if (errno != ENOENT)
         {OssEroute.Emsg("Init", errno, "open", aFname);
          return 0;
         } else opts = O_CREAT|O_TRUNC;
      else if ( buf.st_size != DataSz && buf.st_size)
              {OssEroute.Emsg("Init", aFname, "has invalid size."); return 0;}
              else opts = 0;

// Open the target file
//
   if ((aFD = open(aFname, opts|O_RDWR|O_SYNC, theMode)) < 0)
      {OssEroute.Emsg("Init", errno, "open", aFname);
       return 0;
      }

// Either read the contents or initialize the contents
//
   if (opts & O_CREAT || buf.st_size == 0)
      {memset(uData, 0, sizeof(uData));
       if (!write(aFD, uData, sizeof(uData)))
          {OssEroute.Emsg("Init", errno, "create", aFname);
           return 0;
          }
       nextEnt = 0;
     } else {
      if (!read(aFD, uData, sizeof(uData)))
         {OssEroute.Emsg("Init", errno, "read", aFname);
          return 0;
         }
      for (i = 0; i < maxEnt; i++) if (*uData[i].gName == '\0') break;
      if (i >= maxEnt)
         {OssEroute.Emsg("Init", aFname, "is full.");
          return 0;
         }
      nextEnt = i;
     }

// All done
//
   sprintf(buff, "%d usage log entries in use; %d available.", 
                 nextEnt, maxEnt-nextEnt);
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
  char cgroup[16], *val;
  int qFD, NoGo = 0;

// See if the file has changed (note the firs time through it will have)
//
   if (stat(QFile,&buf))
      {OssEroute.Emsg("Quotas", errno, "process quota file", QFile);
       return 0;
      }
   if (buf.st_mtime == lastMtime) return 0;
   lastMtime = buf.st_mtime;

// Try to open the quota file.
//
   if ( (qFD = open(QFile, O_RDONLY, 0)) < 0)
      {OssEroute.Emsg("Quotas", errno, "open quota file", QFile);
       return 0;
      }

// Attach tyhe file to a stream and tell people what we are doing
//
   OssEroute.Emsg("Quotas", "Processing quota file", QFile);
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
