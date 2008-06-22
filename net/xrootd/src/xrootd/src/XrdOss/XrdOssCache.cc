/******************************************************************************/
/*                                                                            */
/*                        X r d O s s C a c h e . c c                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOssCacheCVSID = "$Id$";

#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <strings.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOss/XrdOssTrace.hh"
  
/******************************************************************************/
/*            G l o b a l s   a n d   S t a t i c   M e m b e r s             */
/******************************************************************************/
  
extern XrdOssSys *XrdOssSS;

extern XrdSysError OssEroute;

extern XrdOucTrace OssTrace;

XrdOssCache_Group *XrdOssCache_Group::fsgroups = 0;

long long          XrdOssCache_Group::PubQuota = -1;

/******************************************************************************/
/*            X r d O s s C a c h e _ F S D a t a   M e t h o d s             */
/******************************************************************************/
  
XrdOssCache_FSData::XrdOssCache_FSData(const char *fsp, 
                                       STATFS_t   &fsbuff,
                                       dev_t       fsID)
{

     path = strdup(fsp);
     size = (long long)fsbuff.f_blocks*fsbuff.FS_BLKSZ;
     frsz = (long long)fsbuff.f_bavail*fsbuff.FS_BLKSZ;
     if (frsz > XrdOssSS->fsFree) XrdOssSS->fsFree = frsz;
     fsid = fsID;
     updt = time(0);
     next = 0;
     stat = 0;
}
  
/******************************************************************************/
/*            X r d O s s C a c h e _ F S   C o n s t r u c t o r             */
/******************************************************************************/

// Cache_FS objects are only created during configuration. No locks are needed.
  
XrdOssCache_FS::XrdOssCache_FS(int &retc,
                               const char *fsGrp,
                               const char *fsPath,
                               FSOpts      fsOpts)
{
   static const mode_t theMode = S_IRWXU | S_IRWXG;
   STATFS_t fsbuff;
   struct stat sfbuff;
   XrdOssCache_FSData *fdp;
   XrdOssCache_FS     *fsp;

// Prefill in case of failure
//
   path = group = 0;

// Verify that this is not a duplicate
//
   fsp = XrdOssSS->fsfirst;
   while(fsp && (strcmp(fsp->path,fsPath)||strcmp(fsp->fsgroup->group,fsGrp)))
        if ((fsp = fsp->next) == XrdOssSS->fsfirst) {fsp = 0; break;}
   if (fsp) {retc = EEXIST; return;}

// Set the groupname and the path which is the supplied path/group name
//
   if (!(fsOpts & isXA)) path = strdup(fsPath);
      else {path = XrdOssPath::genPath(fsPath, fsGrp, suffix);
            if (mkdir(path, theMode) && errno != EEXIST) {retc=errno; return;}
           }
   plen   = strlen(path);
   group  = strdup(fsGrp); 
   fsgroup= 0;
   opts   = fsOpts;
   retc   = ENOMEM;

// Find the filesystem for this object
//
   if (FS_Stat(fsPath, &fsbuff) || stat(fsPath, &sfbuff)) {retc=errno; return;}

// Find the matching filesystem data
//
   fdp = XrdOssSS->fsdata;
   while(fdp) {if (fdp->fsid == sfbuff.st_dev) break; fdp = fdp->next;}

// If we didn't find the filesystem, then create one
//
   if (!fdp)
      {if (!(fdp = new XrdOssCache_FSData(fsPath,fsbuff,sfbuff.st_dev))) return;
          else {fdp->next = XrdOssSS->fsdata; XrdOssSS->fsdata = fdp;}
      }

// Complete the filesystem block (failure now is not an option)
//
   fsdata = fdp;
   retc   = 0;

// Link this filesystem into the filesystem chain
//
   if (!XrdOssSS->fsfirst) {next = this;              XrdOssSS->fscurr = this;
                            XrdOssSS->fsfirst = this; XrdOssSS->fslast = this;
                           }
      else {next = XrdOssSS->fslast->next; XrdOssSS->fslast->next = this;
                   XrdOssSS->fslast = this;
           }

// Check if this is the first group allocation
//
   fsgroup = XrdOssCache_Group::fsgroups;
   while(fsgroup && strcmp(group, fsgroup->group)) fsgroup = fsgroup->next;
   if (!fsgroup && (fsgroup = new XrdOssCache_Group(group, this)))
      {fsgroup->next = XrdOssCache_Group::fsgroups; 
       XrdOssCache_Group::fsgroups=fsgroup;
      }
}

/******************************************************************************/
/*                             f r e e S p a c e                              */
/******************************************************************************/

long long XrdOssCache_FS::freeSpace(long long &Size, const char *path)
{
   STATFS_t fsbuff;
   long long fSpace;

// Free space for a specific path
//
   if (path)
      {if (FS_Stat(path, &fsbuff)) return -1;
       Size = (long long)fsbuff.f_blocks*fsbuff.FS_BLKSZ;
       return fsbuff.f_bavail*fsbuff.FS_BLKSZ;
      }

// Free space for the whole system
//
   XrdOssSS->CacheContext.Lock();
   fSpace = XrdOssSS->fsFree;
   Size   = XrdOssSS->fsSize;
   XrdOssSS->CacheContext.UnLock();
   return fSpace;
}

/******************************************************************************/
/*                                A d j u s t                                 */
/******************************************************************************/
  
void  XrdOssSys::Adjust(dev_t devid, off_t size)
{
   EPNAME("Adjust")
   XrdOssCache_FSData *fsdp;
   XrdOssCache_Group  *fsgp;

// Search for matching filesystem
//
   fsdp = fsdata;
   while(fsdp && fsdp->fsid != devid) fsdp = fsdp->next;
   if (!fsdp) {DEBUG("dev " <<devid <<" not found."); return;}

// Find the public cache group (it might not exist)
//
   fsgp = XrdOssCache_Group::fsgroups;
   while(fsgp && strcmp("public", fsgp->group)) fsgp = fsgp->next;

// Process the result
//
   if (fsdp) 
      {DEBUG("free=" <<fsdp->frsz <<'-' <<size <<" path=" <<fsdp->path);
       CacheContext.Lock();
       if (        (fsdp->frsz  -= size) < 0) fsdp->frsz = 0;
       fsdp->stat |= XrdOssFSData_ADJUSTED;
       if (fsgp && (fsgp->Usage += size) < 0) fsgp->Usage = 0;
       CacheContext.UnLock();
      } else {
       DEBUG("dev " <<devid <<" not found.");
      }
}

/******************************************************************************/
  
void XrdOssSys::Adjust(const char *Path, off_t size, struct stat *buf)
{
   EPNAME("Adjust")
   XrdOssCache_FS *fsp;

// If we have a struct then we need to do some more work
//
   if (buf) 
      {if ((buf->st_mode & S_IFMT) != S_IFLNK) Adjust(buf->st_dev, size);
          else {char lnkbuff[PATH_MAX+64];
                int  lnklen = readlink(Path, lnkbuff, sizeof(lnkbuff)-1);
                if (lnklen > 0)
                   {XrdOssPath::Trim2Base(lnkbuff+lnklen-1);
                    Adjust(lnkbuff, size);
                   }
               }
       return;
      }

// Search for matching logical partition
//
   fsp = fsfirst;
   while(fsp && strcmp(fsp->path, Path)) 
        if ((fsp = fsp->next) == fsfirst) {fsp = 0; break;}

// Process the result
//
   if (fsp) Adjust(fsp, size);
      else {DEBUG("cahe path " <<Path <<" not found.");}
}

/******************************************************************************/
  
void XrdOssSys::Adjust(XrdOssCache_FS *fsp, off_t size)
{
   EPNAME("Adjust")
   XrdOssCache_FSData *fsdp = fsp->fsdata;

// Process the result
//
   if (fsp) 
      {DEBUG("used=" <<fsp->fsgroup->Usage <<'+' <<size <<" path=" <<fsp->path);
       DEBUG("free=" <<fsdp->frsz <<'-' <<size <<" path=" <<fsdp->path);
       CacheContext.Lock();
       if ((fsp->fsgroup->Usage += size) < 0) fsp->fsgroup->Usage = 0;
       if (        (fsdp->frsz  -= size) < 0) fsdp->frsz = 0;
       fsdp->stat |= XrdOssFSData_ADJUSTED;
       if (XrdOssSS->Space) XrdOssSS->Space->Adjust(fsp->fsgroup->GRPid, size);
       CacheContext.UnLock();
      }
}

/******************************************************************************/
/*                            F i n d _ C a c h e                             */
/******************************************************************************/
  
XrdOssCache_FS *XrdOssSys::Find_Cache(const char *Path)
{
   XrdOssCache_FS *fsp;
   char lnkbuff[PATH_MAX+64];
   struct stat sfbuff;
   int lnklen;

// First see if this is a symlink that refers to a new style cache
//
   if (lstat(Path, &sfbuff)
   ||  (sfbuff.st_mode & S_IFMT) != S_IFLNK
   ||  (lnklen = readlink(Path, lnkbuff, sizeof(lnkbuff)-1)) <= 0
   ||  lnkbuff[lnklen-1] != XrdOssPath::xChar) return 0;

// Trim the link to the base name
//
   XrdOssPath::Trim2Base(lnkbuff+lnklen-1);

// Search for matching logical partition
//
   fsp = fsfirst;
   while(fsp && strcmp(fsp->path, lnkbuff))
        if ((fsp = fsp->next) == fsfirst) {fsp = 0; break;}
   return fsp;
}

/******************************************************************************/
/*                            L i s t _ C a c h e                             */
/******************************************************************************/
  
void XrdOssSys::List_Cache(const char *lname, XrdSysError &Eroute)
{
     XrdOssCache_FS *fsp;
     const char *theOpt;
     char *pP, buff[4096];

     if ((fsp = fsfirst)) do
        {if (fsp->opts & XrdOssCache_FS::isXA)
            {pP = (char *)fsp->path + fsp->plen - 1;
             do {pP--;} while(*pP != '/');
             *pP = '\0';   theOpt = " xa";
            } else {pP=0;  theOpt = "";}
         snprintf(buff, sizeof(buff), "%s %s %s%s", lname, 
                        fsp->group, fsp->path, theOpt);
         if (pP) *pP = '/';
         Eroute.Say(buff);
         fsp = fsp->next;
        } while(fsp != fsfirst);
}
 
/******************************************************************************/
/*                             C a c h e S c a n                              */
/******************************************************************************/

void *XrdOssSys::CacheScan(void *carg)
{
   EPNAME("CacheScan")
   XrdOssCache_FSData *fsdp;
   STATFS_t fsbuff;
   const struct timespec naptime = {XrdOssSS->cscanint, 0};
   int retc;

// Loop scanning the cache
//
   while(1)
        {nanosleep(&naptime, 0);

        // Get the cache context lock
        //
           XrdOssSS->CacheContext.Lock();

        // Scan through all filesystems skip filesystem that have been
        // recently adjusted to avoid fs statstics latency problems.
        //
           XrdOssSS->fsSize =  0;
           fsdp = XrdOssSS->fsdata;
           while(fsdp)
                {retc = 0;
                 if ((fsdp->stat & XrdOssFSData_REFRESH)
                 || !(fsdp->stat & XrdOssFSData_ADJUSTED))
                     {if ((retc = FS_Stat(fsdp->path, &fsbuff)))
                         OssEroute.Emsg("XrdOssCacheScan", errno ,
                                    "state file system ",(char *)fsdp->path);
                         else {fsdp->frsz = fsbuff.f_bavail*fsbuff.FS_BLKSZ;
                               fsdp->stat &= ~(XrdOssFSData_REFRESH |
                                               XrdOssFSData_ADJUSTED);
                               DEBUG("New free=" <<fsdp->frsz <<" path=" <<fsdp->path);
                               }
                     } else fsdp->stat |= XrdOssFSData_REFRESH;
                 if (!retc && fsdp->frsz > XrdOssSS->fsFree)
                    {XrdOssSS->fsFree = fsdp->frsz;
                     XrdOssSS->fsSize = fsdp->size;
                    }
                 fsdp = fsdp->next;
                }

         // Unlock the cache and if we have quotas check them out
         //
            XrdOssSS->CacheContext.UnLock();
            if (Quotas) Quotas->Quotas();
         }

// Keep the compiler happy
//
   return (void *)0;
}

/******************************************************************************/
/*                               R e C a c h e                                */
/******************************************************************************/

// Recache() is only called during configuration and no locks are needed.
  
int XrdOssSys::ReCache(const char *Path, const char *Qfile)
{
     XrdOssCache_Group *cgp;
     XrdOssSpace *sP;
     long long bytesUsed;

// If usage directory or quota file was passed then we initialize space handling
// We need to create a space object to track usage across failures.
//
   if (Path || Qfile)
      {sP = new XrdOssSpace(Path, Qfile);
       if (!sP->Init()) return 1;
       if (Path)  Space = sP;
       if (Qfile) Quotas = sP;
      }

// If we will be saving space information then we need to assign each group
// to a save set. If there is no space object then skip all of this.
//
   if (Path && (cgp = XrdOssCache_Group::fsgroups))
      do {cgp->GRPid = Space->Assign(cgp->group, bytesUsed);
          cgp->Usage = bytesUsed;
         } while((cgp = cgp->next));
   return 0;
}
