/******************************************************************************/
/*                                                                            */
/*                       X r d O s s C r e a t e . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
/******************************************************************************/
/*                             i n c l u d e s                                */
/******************************************************************************/
  
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <strings.h>
#include <stdio.h>
#include <utime.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/param.h>
#if defined(__solaris__) || defined(AIX)
#include <sys/vnode.h>
#endif

#include "XrdFrm/XrdFrmXAttr.hh"
#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssCopy.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOss/XrdOssTrace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucExport.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdOuc/XrdOucXAttr.hh"
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
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
class XrdOssCreateInfo
     {public:
      unsigned long long pOpts;
      const char        *Path;
      mode_t             Amode;
      int                cOpts;
      XrdOssCreateInfo(const char *path, mode_t amode, int opts)
                      : Path(path), Amode(amode), cOpts(opts) {}
     ~XrdOssCreateInfo() {}
     };

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
    const int AMode = S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH; // 775
    char  local_path[MAXPATHLEN+1], *p, pc;
    unsigned long long remotefs;
    int isLink = 0, Missing = 1, retc = 0, datfd;
    XrdOssCreateInfo crInfo(local_path, access_mode, Opts);
    struct stat buf;

// Get options associated with this path and check if it's r/w
//
   remotefs = Check_RO(Create, crInfo.pOpts, path, "create");

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
      return XrdOssSS->Stage(tident, path, env, Opts>>8, 
                             access_mode, crInfo.pOpts);

// The file must not exist if it's declared "new". Otherwise, reuse the space.
// SetFattr() alaways closes the provided file descriptor!
//
   if (!Missing)
      {if (Opts & XRDOSS_new)                 return -EEXIST;
       if ((buf.st_mode & S_IFMT) == S_IFDIR) return -EISDIR;
       do {datfd = open(local_path, Opts>>8, access_mode);}
                   while(datfd < 0 && errno == EINTR);
       if (datfd < 0) return -errno;
       if ((retc = SetFattr(crInfo, datfd, buf.st_mtime))) return retc;
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

     // Create the file in remote system unless not wanted so
     //
        if (crInfo.pOpts & XRDEXP_RCREATE)
           {if ((retc = MSS_Create(remote_path, access_mode, env)) < 0)
               {DEBUG("rc" <<retc <<" mode=" <<std::oct <<access_mode
                           <<std::dec <<" remote path=" <<remote_path);
                return retc;
               }
           } else if (!(crInfo.pOpts & XRDEXP_NOCHECK))
                     {if (!(retc = MSS_Stat(remote_path))) return -EEXIST;
                         else if (retc != -ENOENT)         return retc;
                     }
      }

// Created file in the extended cache or the local name space
//
   if (XrdOssCache::fsfirst && !(crInfo.pOpts & XRDEXP_INPLACE))
           retc = Alloc_Cache(crInfo, env);
      else retc = Alloc_Local(crInfo, env);

// If successful then check if xattrs were actually set
//
   if (retc == XrdOssOK && crInfo.cOpts & XRDOSS_setnoxa)
      {XrdOucPList *plP = RPList.About(path);
       if (plP) plP->Set(plP->Flag() | XRDEXP_NOXATTR);
      }

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

int XrdOssSys::Alloc_Cache(XrdOssCreateInfo &crInfo, XrdOucEnv &env)
{
   EPNAME("Alloc_Cache")
   int datfd, rc;
   char pbuff[MAXPATHLEN+1], cgbuff[XrdOssSpace::minSNbsz], *tmp;
   XrdOssCache::allocInfo aInfo(crInfo.Path, pbuff, sizeof(pbuff));

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
   aInfo.aMode  = crInfo.Amode;
   if ((datfd = XrdOssCache::Alloc(aInfo)) < 0) return datfd;

// Set the pfn as the extended attribute if we are in new mode
//
   if (!runOld && !(crInfo.pOpts & XRDEXP_NOXATTR)
   &&  (rc = XrdSysFAttr::Set(XrdFrmXAttrPfn::Name(), crInfo.Path,
                              strlen(crInfo.Path)+1, pbuff, datfd))) return rc;

// Set extended attributes for this newly created file if allowed to do so.
// SetFattr() alaways closes the provided file descriptor!
//
   if ((rc = SetFattr(crInfo, datfd, 1))) return rc;

// Now create a symbolic link to the target
//
   if ((symlink(pbuff, crInfo.Path) && errno != EEXIST)
   ||  unlink(crInfo.Path) || symlink(pbuff, crInfo.Path))
      {rc = -errno; unlink(pbuff);}

// Now create a symlink from the cache pfn to the actual path (xa runOld only)
//
   if (runOld && aInfo.cgPsfx)
      {strcpy(aInfo.cgPsfx, ".pfn");
       if ((symlink(crInfo.Path, pbuff) && errno != EEXIST)
       ||  unlink(pbuff) || symlink(crInfo.Path, pbuff)) rc = -errno;
       *(aInfo.cgPsfx) = '\0';
       if (rc) {unlink(pbuff); unlink(crInfo.Path);}
      }

// All done
//
   DEBUG(aInfo.cgName <<" cache for " <<pbuff);
   return rc;
}

/******************************************************************************/
/*                           A l l o c _ L o c a l                            */
/******************************************************************************/

int XrdOssSys::Alloc_Local(XrdOssCreateInfo &crInfo, XrdOucEnv &env)
{
   int datfd, rc;

// Simply open the file in the local filesystem, creating it if need be.
//
   do {datfd = open(crInfo.Path, O_CREAT|O_TRUNC, crInfo.Amode);}
               while(datfd < 0 && errno == EINTR);
   if (datfd < 0) return -errno;

// Set extended attributes for this newly created file if allowed to do so.
// SetFattr() alaways closes the provided file descriptor!
//
   if ((rc = SetFattr(crInfo, datfd, 1))) return rc;

// All done
//
   return XrdOssOK;
}
  
/******************************************************************************/
/*                              S e t F a t t r                               */
/******************************************************************************/
  
int XrdOssSys::SetFattr(XrdOssCreateInfo &crInfo, int fd, time_t mtime)
{
   static const char *lkSuffix = ".lock";
   static const int   lkSuffsz = 5;

   class  fdCloser
         {public:
          const char *Path;
          int         theFD;
          int         Done(int rc) {if (rc) unlink(Path); return rc;}
                      fdCloser(const char *pn, int fd) : Path(pn), theFD(fd) {}
                      fdCloser() {close(theFD);}
         } Act(crInfo.Path, fd);

   XrdOucXAttr<XrdFrmXAttrCpy> crX;
   int rc;

// Skip all of this if we do not need to create a lock file
//
   if (!(XRDEXP_MAKELF & crInfo.pOpts)) return Act.Done(0);

// If we are running in backward compatability mode, then we need to create
// an old-style lock file.
//
   if (runOld)
      {struct utimbuf times;
       char lkBuff[MAXPATHLEN+lkSuffsz+1];
       int  lkfd, n = strlen(crInfo.Path);
       if (n+lkSuffsz >= (int)sizeof(lkBuff))
          return Act.Done(OssEroute.Emsg("Create", -ENAMETOOLONG,
                                         "generate lkfname for", crInfo.Path));
       strcpy(lkBuff, crInfo.Path); strcpy(lkBuff+n, lkSuffix);
       do {lkfd = open(lkBuff, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);}
          while( lkfd < 0 && errno == EINTR);
       if ( lkfd < 0)
          return Act.Done(OssEroute.Emsg("Create", -errno, "create", lkBuff));
       close(lkfd); times.actime = time(0); times.modtime = mtime;
       if (utime(lkBuff, (const struct utimbuf *)&times))
          return Act.Done(OssEroute.Emsg("Create",-errno,"set mtime for",lkBuff));
       return Act.Done(0);
      }

// Check if we should really create any extended attribute
//
   if ((crInfo.pOpts & XRDEXP_NOXATTR)) return Act.Done(0);

// Set copy time
//
   crX.Attr.cpyTime = static_cast<long long>(mtime);
   rc = crX.Set(crInfo.Path, fd);

// Check if extended attribute were set and indicate whether it is supported
//
   if (rc == -ENOTSUP) {rc = 0; crInfo.cOpts |= XRDOSS_setnoxa;}
   return Act.Done(rc);
}
