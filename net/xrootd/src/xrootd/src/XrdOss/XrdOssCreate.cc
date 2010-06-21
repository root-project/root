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
#include <sys/param.h>
#if defined(__solaris__) || defined(AIX)
#include <sys/vnode.h>
#endif

#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssCopy.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssLock.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOss/XrdOssTrace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucExport.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
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
    char  local_path[MAXPATHLEN+1], *p, pc;
    unsigned long long popts, remotefs;
    int isLink = 0, Missing = 1, retc = 0, datfd;
    XrdOssLock path_dir, new_file;
    struct stat buf;

// Get options associated with this path and check if it's r/w
//
   remotefs = Check_RO(Create, popts, path, "create");

// Generate the actual local path for this file.
//
   if ((retc = GenLocalPath(path, local_path))) return retc;

// Determine the state of the file. We will need this information as we go on.
//
   if ((Missing = lstat(local_path, &buf))) retc = errno;
      else {if ((isLink = ((buf.st_mode & S_IFMT) == S_IFLNK)))
               {if (stat(local_path, &buf))
                   {if (errno != ENOENT) return -errno;
                    OssEroute.Emsg("Create","removing dangling link",local_path);
                    if (unlink(local_path)) retc = errno;
                    Missing = 1; isLink = 0;
                   }
               }
            }
   if (retc && retc != ENOENT) return -retc;

// At this point, creation requests may need to be routed via the stagecmd.
// This is done if the file/link do not exist. Otherwise, we drop through.
//
   if (StageCreate && Missing)
      return XrdOssSS->Stage(tident, path, env, Opts>>8, access_mode, popts);

// The file must not exist if it's declared "new". Otherwise, reuse the space.
//
   if (!Missing)
      {if (Opts & XRDOSS_new)                 return -EEXIST;
       if ((buf.st_mode & S_IFMT) == S_IFDIR) return -EISDIR;
       do {datfd = open(local_path, Opts>>8, access_mode);}
                   while(datfd < 0 && errno == EINTR);
           if (datfd < 0) return -errno;
       close(datfd);
       if (Opts>>8 & O_TRUNC && buf.st_size)
          {off_t theSize = buf.st_size;
           if (isLink) {buf.st_mode = (buf.st_mode & ~S_IFMT) | S_IFLNK;
                        XrdOssCache::Adjust(local_path, -theSize, &buf);
                       }
          }
       return 0;
      }

// If the path is to be created, make sure the path exists at this point
//
   if ((Opts & XRDOSS_mkpath) && (p = rindex(local_path, '/')))
      {p++; pc = *p; *p = '\0';
       XrdOucUtils::makePath(local_path, AMode);
       *p = pc;
      }

// If this is a staging filesystem then we have lots more work to do.
//
   if (remotefs)
      {char remote_path[MAXPATHLEN+1];

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
                     {if (!(retc = MSS_Stat(remote_path)))
                         {path_dir.UnSerialize(0); return -EEXIST;}
                         else if (retc != -ENOENT)
                                 {path_dir.UnSerialize(0); return retc;}
                     }
      }

// Created file in the extended cache or the local name space
//
   if (XrdOssCache::fsfirst && !(popts & XRDEXP_INPLACE))
           retc = Alloc_Cache(local_path, access_mode, env);
      else retc = Alloc_Local(local_path, access_mode, env);

// If successful, appropriately manage the locks.
//
   if (retc == XrdOssOK && (popts & XRDEXP_MAKELF))
      if (new_file.Serialize(local_path,LKFlags) >= 0) new_file.UnSerialize(0);

// All done.
//
   if (remotefs) path_dir.UnSerialize(0);
   return retc;
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                           A l l o c _ C a c h e                            */
/******************************************************************************/

int XrdOssSys::Alloc_Cache(const char *path, mode_t amode, XrdOucEnv &env)
{
   EPNAME("Alloc_Cache")
   int datfd;
   char pbuff[MAXPATHLEN+1], cgbuff[XrdOssSpace::minSNbsz], *tmp;
   XrdOssCache::allocInfo aInfo(path, pbuff, sizeof(pbuff));

// Grab the suggested size from the environment
//
   if ((tmp = env.Get(OSS_ASIZE))
   &&  !XrdOuca2x::a2ll(OssEroute,"invalid asize",tmp,&aInfo.cgSize,0))
      return -XRDOSS_E8018;

// Get the correct cache group and partition path
//
   if ((aInfo.cgPath=XrdOssCache::Parse(env.Get(OSS_CGROUP),cgbuff,sizeof(cgbuff))))
      aInfo.cgPlen = strlen(aInfo.cgPath);

// Allocate space in the cache.
//
   aInfo.cgName = cgbuff;
   aInfo.aMode  = amode;
   if ((datfd = XrdOssCache::Alloc(aInfo)) < 0) return datfd;
   close(datfd);

// Now create a symbolic link to the target
//
   if ((symlink(pbuff, path) && errno != EEXIST)
   ||  unlink(path) || symlink(pbuff, path)) {datfd = -errno; unlink(pbuff);}

// Now create a symlink from the cache pfn to the actual path (xa only)
//
   if (aInfo.cgPsfx)
      {strcpy(aInfo.cgPsfx, ".pfn");
       if ((symlink(path, pbuff) && errno != EEXIST)
       ||  unlink(pbuff) || symlink(path, pbuff)) datfd = -errno;
       *(aInfo.cgPsfx) = '\0';
       if (datfd < 0) {unlink(pbuff); unlink(path);}
      }

// All done
//
   DEBUG(aInfo.cgName <<" cache for " <<pbuff);
   return (datfd >= 0 ? XrdOssOK : datfd);
}

/******************************************************************************/
/*                           A l l o c _ L o c a l                            */
/******************************************************************************/

int XrdOssSys::Alloc_Local(const char *path, mode_t amode, XrdOucEnv &env)
{
   int datfd;

// Simply open the file in the local filesystem, creating it if need be.
//
   do {datfd = open(path, O_CREAT|O_TRUNC, amode);}
               while(datfd < 0 && errno == EINTR);
   if (datfd < 0) return -errno;
   close(datfd);
   return XrdOssOK;
}
