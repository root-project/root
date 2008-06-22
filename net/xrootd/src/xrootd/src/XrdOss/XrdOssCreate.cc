/******************************************************************************/
/*                                                                            */
/*                       X r d O s s C r e a t e . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdOssCreateCVSID = "$Id$";

/******************************************************************************/
/*                             i n c l u d e s                                */
/******************************************************************************/
  
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <strings.h>
#include <stdio.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#if defined(__solaris__) || defined(AIX)
#include <sys/vnode.h>
#endif

#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssLock.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssTrace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOuc/XrdOucExport.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                  E r r o r   R o u t i n g   O b j e c t                   */
/******************************************************************************/
  
extern XrdSysError OssEroute;

extern XrdOucTrace OssTrace;

extern XrdOssSys  *XrdOssSS;

/******************************************************************************/
/*                                c r e a t e                                 */
/******************************************************************************/

/*
  Function: Create a file named `path' with 'file_mode' access mode bits set.

  Input:    path        - The fully qualified name of the file to create.
            access_mode - The Posix access mode bits to be assigned to the file.
                          These bits correspond to the standard Unix permission
                          bits (e.g., 744 == "rwxr--r--").
            env         - Environmental information.
            opts        - Set as follows:
                          XRDOSS_mkpath - create dir path if it does not exist.
                          XRDOSS_new    - the file must not already exist.
                          x00000000     - x are standard open flags (<<8)

  Output:   Returns XRDOSS_OK upon success; (-errno) otherwise.
*/
int XrdOssSys::Create(const char *tident, const char *path, mode_t access_mode,
                      XrdOucEnv &env, int Opts)
{
    EPNAME("Create")
    const int LKFlags = XrdOssFILE|XrdOssSHR|XrdOssNOWAIT|XrdOssRETIME;
    const int AMode = S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH; // 775
    char  local_path[XrdOssMAX_PATH_LEN+1], *p;
    char remote_path[XrdOssMAX_PATH_LEN+1];
    unsigned long long popts, remotefs;
    int retc, datfd;
    XrdOssLock path_dir, new_file;
    struct stat buf;

// Determine whether we can actually create a file on this server.
//
   remotefs = Check_RO(Create, popts, path, "creating ");

// Generate the actual local path for this file.
//
   if ((retc = GenLocalPath(path, local_path))) return retc;

// At this point, creation requests may need to be routed via the stagecmd
// which will take care of all the stuff we do here.
//
   if (StageCreate)
      {struct stat buf;
       if (lstat(local_path, &buf))
          {if (errno != ENOENT) return -errno;
              else return XrdOssSS->Stage(tident,path,env,Opts>>8,access_mode);
          }
       return 0;
      }

// The file must not exist if it's declared "new". Otherwise, we must reuse the
// file, especially in the presence of multiple filesystems
//
   if (!stat(local_path, &buf))
      {if (Opts & XRDOSS_new)     return -EEXIST;
       if (buf.st_mode & S_IFDIR) return -EISDIR;
       do {datfd = open(local_path, Opts>>8, access_mode);}
                   while(datfd < 0 && errno == EINTR);
           if (datfd < 0) return -errno;
       close(datfd);
       if (Opts>>8 & O_TRUNC && buf.st_size)
          {off_t theSize = buf.st_size;
           if (!lstat(local_path, &buf)) Adjust(local_path, -theSize, &buf);
          }
       return 0;
      }

// If the path is to be created, make sure the path exists at this point
//
   if ((Opts & XRDOSS_mkpath) && (retc = strlen(local_path)))
      {if (local_path[retc-1] == '/') local_path[retc-1] = '\0';
       if ((p = rindex(local_path, int('/'))) && p != local_path)
          {*p = '\0';
           if (stat(local_path, &buf) && errno == ENOENT)
              Mkpath(local_path, AMode);
           *p = '/';
          }
      }

// If this is a staging filesystem then we have lots more work to do.
//
   if (remotefs)
      {
      // Generate the remote path for this file
      //
         if ((retc = GenRemotePath(path,remote_path))) return retc;

      // Gain exclusive control over the directory.
      //
         if ( (retc = path_dir.Serialize(local_path, XrdOssDIR|XrdOssEXC)) < 0)
            return retc;

     // Create the file in remote system unless not wanted so
     //
        if (popts & XRDEXP_RCREATE)
           {if ((retc = MSS_Create(remote_path, access_mode, env)) < 0)
               {path_dir.UnSerialize(0);
                DEBUG("rc" <<retc <<" mode=" <<std::oct <<access_mode
                           <<std::dec <<" remote path=" <<remote_path);
                return retc;
               }
           } else if (!(popts & XRDEXP_NOCHECK))
                     {if (!(retc = MSS_Stat(remote_path, &buf)))
                         {path_dir.UnSerialize(0); return -EEXIST;}
                         else if (retc != -ENOENT)
                                 {path_dir.UnSerialize(0); return retc;}
                     }
      }

// Created file in the extended cache or the local name space
//
   if (fsfirst && !(popts & XRDEXP_INPLACE))
           datfd = Alloc_Cache(local_path, Opts>>8, access_mode, env);
      else datfd = Alloc_Local(local_path, Opts>>8, access_mode, env);

// If successful, appropriately manage the locks.
//
   if (datfd >= 0)
      {if (remotefs || (popts & XRDEXP_MIG))
          {if ((new_file.Serialize(local_path,LKFlags))
                >= 0) new_file.UnSerialize(0);
           if (remotefs) path_dir.UnSerialize(0);
          }
       close(datfd); retc = XrdOssOK;
      } else         retc = datfd;

// All done.
//
   return retc;
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                           A l l o c _ C a c h e                            */
/******************************************************************************/

int XrdOssSys::Alloc_Cache(const char *path, int Oflag, mode_t amode, 
                           XrdOucEnv &env)
{
   EPNAME("Alloc_Cache")
   static const mode_t theMode = S_IRWXU | S_IRWXG;
   double fuzz, diffree;
   int datfd, rc;
   XrdOssPath::fnInfo Info;
   XrdOssCache_FS *fsp, *fspend, *fsp_sel;
   XrdOssCache_Group *cgp = 0;
   long long size, maxfree, curfree;
   char pbuff[XrdOssMAX_PATH_LEN+1], *cgroup, *vardata, *pfnSfx;

// Grab the suggested size from the environment
//
   if (!(vardata = env.Get(OSS_ASIZE))) size = 0;
      else if (!XrdOuca2x::a2ll(OssEroute,"invalid asize",vardata,&size,0))
              return -XRDOSS_E8018;

// Get the correct cache group and user
//
   if (!(cgroup = env.Get(OSS_CGROUP))) cgroup = OSS_CGROUP_DEFAULT;
   Info.User  = env.Get(SEC_USER);

// Compute appropriate allocation size and fuzz factor
//
   if ( (size = size * ovhalloc / 100 + size) < minalloc) size = minalloc;
   fuzz = static_cast<double>(fuzalloc)/100.0;

// Find the corresponding cache group
//
   cgp = XrdOssCache_Group::fsgroups;
   while(cgp && strcmp(cgroup, cgp->group)) cgp = cgp->next;
   if (!cgp) return -XRDOSS_E8019;

// Get the cache lock and select correct cursor and fuzz amount
//
   CacheContext.Lock();
   fsp = cgp->curr;

// Find a cache that will fit this allocation request
//
   maxfree = fsp->fsdata->frsz; fsp_sel = 0; fspend = fsp = fsp->next;
   do {
       if (strcmp(cgroup, fsp->group)) continue;
       curfree = fsp->fsdata->frsz;
       if (size > curfree) continue;

             if (!fsp_sel)       {fsp_sel = fsp; maxfree = curfree;}
       else  if (!fuzz) {if (curfree > maxfree) 
                                 {fsp_sel = fsp; maxfree = curfree;}}
       else {diffree = (!(curfree + maxfree) ? 0.0
                     : static_cast<double>(::llabs(maxfree - curfree)) /
                       static_cast<double>(        maxfree + curfree));
             if (diffree > fuzz) {fsp_sel = fsp; maxfree = curfree;}
            }
      } while((fsp = fsp->next) != fspend);

// Check if we can realy fit this file
//
   if (!fsp_sel) {CacheContext.UnLock(); return -XRDOSS_E8020;}

// Construct the target filename
//
   Info.Path = fsp_sel->path; Info.Plen = fsp_sel->plen; 
   Info.Sfx  = fsp_sel->suffix;
   pfnSfx = XrdOssPath::genPFN(Info, pbuff, sizeof(pbuff),
                              (fsp->opts & XrdOssCache_FS::isXA ? 0 : path));
   if (!(*pbuff)) return -ENAMETOOLONG;

// Simply open the file in the local filesystem, creating it if need be.
//
   rc = 0;
   do {do {datfd = open(pbuff, Oflag, amode);}
          while(datfd < 0 && errno == EINTR);
       if (datfd < 0 && (errno != ENOENT || pfnSfx == 0)) rc = 1;
          else {*Info.Slash = '\0';
                rc = mkdir(pbuff, theMode);
                *Info.Slash = '/';
               }
      } while(!rc);

// Now create a symlink from the cache pfn to the actual path (xa only)
//
   if (datfd >= 0 && pfnSfx)
      {strcpy(pfnSfx, ".pfn");
       if ((symlink(path, pbuff) && errno != EEXIST)
          || unlink(pbuff) || symlink(path, pbuff))
          {close(datfd); *pfnSfx = '\0'; unlink(pbuff); datfd = -1;}
          else *pfnSfx = '\0';
      }

// Now create a symbolic link to the target and adjust free space
//
   if (datfd < 0) datfd = -errno;
      else if ((symlink(pbuff, path) && errno != EEXIST)
           || unlink(path) || symlink(pbuff, path))
              {rc = -errno; close(datfd); datfd = rc; unlink(pbuff);
               if (pfnSfx) {*pfnSfx = '.'; unlink(pbuff); *pfnSfx = '\0';}
              }
              else fsp_sel->fsdata->frsz -= size;

// Update the cursor address
//
   if (cgp) cgp->curr = fsp_sel->next;
      else fscurr = fsp_sel->next;

// All done
//
   CacheContext.UnLock();
   DEBUG(cgroup <<" cache as " <<pbuff);
   return datfd;
}

/******************************************************************************/
/*                           A l l o c _ L o c a l                            */
/******************************************************************************/
  
int XrdOssSys::Alloc_Local(const char *path, int Oflag, mode_t amode, 
                           XrdOucEnv &env)
{
   int datfd;

// Simply open the file in the local filesystem, creating it if need be.
//
   do {datfd = open(path, Oflag, amode);}
               while(datfd < 0 && errno == EINTR);
   return (datfd < 0 ? -errno : datfd);
}
