/******************************************************************************/
/*                                                                            */
/*                        X r d O s s R e l o c . c c                         */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdOssRelocCVSID = "$Id$";

/******************************************************************************/
/*                             i n c l u d e s                                */
/******************************************************************************/
  
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <strings.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/param.h>

#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssCopy.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOss/XrdOssTrace.hh"
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
/*                                 R e l o c                                  */
/******************************************************************************/

/*
  Function: Relocate/Copy the file at `path' to a new location.

  Input:    path        - The fully qualified name of the file to relocate.
            cgName      - Target space name[:path]
            anchor      - The base path where a symlink to the copied file is
                          to be created. If present, the original file is kept.

  Output:   Returns XrdOssOK upon success; (-errno) otherwise.
*/
int XrdOssSys::Reloc(const char *tident, const char *path,
                     const char *cgName, const char *anchor)
{
   EPNAME("Reloc")
   const int AMode = S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH; // 775
   class pendFiles
        {public:
         char *pbuff;
         char *tbuff;
         int   datfd;
               pendFiles(char *pb, char *tb) : datfd(-1)
                           {pbuff = pb; *pb = '\0';
                            tbuff = tb; *tb = '\0';
                           }
              ~pendFiles() {if (datfd >= 0) close(datfd);
                            if (pbuff && *pbuff) unlink(pbuff);
                            if (tbuff && *tbuff) unlink(tbuff);
                           }
        };
   char cgNow[XrdOssSpace::minSNbsz], cgbuff[XrdOssSpace::minSNbsz];
   char lbuff[MAXPATHLEN+8];
   char pbuff[MAXPATHLEN+8];
   char tbuff[MAXPATHLEN+8];
   char local_path[MAXPATHLEN+8];
   pendFiles PF(pbuff, tbuff);
   XrdOssCache::allocInfo aInfo(path, pbuff, sizeof(pbuff));
   int rc, lblen, datfd;
   struct stat buf;

// Generate the actual local path for this file.
//
   if ((rc = GenLocalPath(path, local_path))) return rc;

// Determine the state of the file.
//
   if (stat(local_path, &buf)) return -errno;
   if ((buf.st_mode & S_IFMT) == S_IFDIR) return -EISDIR;
   if ((buf.st_mode & S_IFMT) != S_IFREG) return -ENOTBLK;

// Get the correct cache group and partition path
//
   if ((aInfo.cgPath = XrdOssCache::Parse(cgName, cgbuff, sizeof(cgbuff))))
      aInfo.cgPlen = strlen(aInfo.cgPath);

// Verify that this file will go someplace other than where it is now
//
   lblen = XrdOssPath::getCname(local_path, cgNow, lbuff, sizeof(lbuff)-7);
   lbuff[lblen] = '\0';
   if (!strcmp(cgbuff, cgNow)
   && (!aInfo.cgPath || !strncmp(aInfo.cgPath, lbuff, aInfo.cgPlen)))
      return -EEXIST;

// Allocate space in the cache. Note that the target must be an xa cache
//
   aInfo.aMode  = buf.st_mode & S_IAMB;
   aInfo.cgSize = buf.st_size;
   aInfo.cgName = cgbuff;
   if ((PF.datfd = datfd = XrdOssCache::Alloc(aInfo)) < 0) return datfd;
   if (!aInfo.cgPsfx) return -ENOTSUP;

// Copy the original file to the new location
//
   if (XrdOssCopy::Copy(path, pbuff, datfd) < 0) return -EIO;
   close(datfd); PF.datfd = -1;

// If the file is to be merely copied, substitute the desired destination
//
   if (!anchor) {strcpy(tbuff, local_path); strcat(tbuff, ".anew");}
      else {struct stat sbuf;
            char *Slash;
            if (strlen(anchor)+strlen(path) >= sizeof(local_path))
               return -ENAMETOOLONG;
            strcpy(local_path, anchor); strcat(local_path, path);
            if (!(Slash = rindex(local_path, '/'))) return -ENOTDIR;
            *Slash = '\0'; rc = stat(local_path, &sbuf); *Slash = '/';
            if (rc && (rc = XrdOucUtils::makePath(local_path, AMode)))
               return rc;
            strcpy(tbuff, local_path);
           }

// Now create a symbolic link to the target
//
   if ((symlink(pbuff, tbuff) && errno != EEXIST)
   || unlink(tbuff) || symlink(pbuff, tbuff)) return -errno;

// Rename the link atomically over the existing name
//
   if (!anchor && rename(tbuff, local_path) < 0) return -errno;
   PF.tbuff = 0; PF.pbuff = 0;

// Now create a symlink from the cache pfn to the actual path (xa only)
//
   strcpy(aInfo.cgPsfx, ".pfn"); rc = 0;
   if ((symlink(local_path, pbuff) && errno != EEXIST)
   || unlink(pbuff) || symlink(local_path, pbuff)) rc = errno;

// Issue warning if the pfn file could not be created (very very rare).
// At this point we can't do much about it.
//
   if (rc) OssEroute.Emsg("Reloc", rc, "create symlink", pbuff);
   *(aInfo.cgPsfx) = '\0';

// If this was a copy operation, we are done
//
   DEBUG(cgNow <<':' <<local_path <<" -> " <<aInfo.cgName <<':' <<pbuff);
   if (anchor) return XrdOssOK;

// Check if the original file was a symlink and that has to be deleted
// Adjust the space usage numbers at this point as well.
//
   if (*lbuff)
      {if (unlink(lbuff))     OssEroute.Emsg("Reloc",errno,"removing",lbuff);
       if (XrdOssPath::isXA(lbuff))
          {strcat(lbuff, ".pfn");
           if (unlink(lbuff)) OssEroute.Emsg("Reloc",errno,"removing",lbuff);
          }
       XrdOssCache::Adjust(XrdOssCache::Find(lbuff, lblen), -buf.st_size);
       } else XrdOssCache::Adjust(buf.st_dev, -buf.st_size);

// All done (permanently adjust usage for the target)
//
   XrdOssCache::Adjust(aInfo.cgFSp, buf.st_size);
   return XrdOssOK;
}
