/******************************************************************************/
/*                                                                            */
/*                       X r d O s s U n l i n k . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOssUnlinkCVSID = "$Id$";

#include <unistd.h>
#include <errno.h>
#include <strings.h>
#include <limits.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssLock.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssTrace.hh"

/******************************************************************************/
/*           G l o b a l   E r r o r   R o u t i n g   O b j e c t            */
/******************************************************************************/

extern XrdSysError OssEroute;

extern XrdOucTrace OssTrace;
  
/******************************************************************************/
/*                                R e m d i r                                 */
/******************************************************************************/
  
/*
  Function: Delete a directory from the namespace.

  Input:    path      - Is the fully qualified name of the dir to be removed.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/
int XrdOssSys::Remdir(const char *path, int Opts)
{
    unsigned long long opts;
    int retc;
    struct stat statbuff;
    char  local_path[MAXPATHLEN+1+8];

// Build the right local and remote paths.
//
   if (Opts & XRDOSS_isPFN) strcpy(local_path, path);
      else {retc = Check_RO(Unlink, opts, path, "deleting ");
            if ( (retc = GenLocalPath( path,  local_path))) return retc;
           }

// Check if this path is really a directory
//
    if (lstat(local_path, &statbuff)) return (errno == ENOENT ? 0 : -errno);
    if ((statbuff.st_mode & S_IFMT) != S_IFDIR) return -ENOTDIR;

// Complete by calling Unlink()
//
    return Unlink(path, Opts);
}

/******************************************************************************/
/*                                U n l i n k                                 */
/******************************************************************************/

/*
  Function: Delete a file from the namespace and release it's data storage.

  Input:    path      - Is the fully qualified name of the file to be removed.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/
int XrdOssSys::Unlink(const char *path, int Opts)
{
    EPNAME("Unlink")
    unsigned long long haslf, remotefs;
    int i, retc2, doAdjust = 0, retc = XrdOssOK;
    XrdOssLock un_file;
    struct stat statbuff;
    char *fnp;
    char  local_path[MAXPATHLEN+1+8];
    char remote_path[MAXPATHLEN+1];

// Build the right local and remote paths.
//
   if (Opts & XRDOSS_isPFN)
      {strcpy(local_path, path),
       *remote_path = '\0';
       haslf = Opts & XRDOSS_isMIG;
       remotefs = 0;
      } else {
       remotefs = Check_RO(Unlink, haslf, path, "deleting ");
       if ( (retc = GenLocalPath( path,  local_path))
       ||   (retc = GenRemotePath(path, remote_path)) ) return retc;
       haslf &= XRDEXP_MAKELF;
      }

 // Serialize the directory.
 //
    if (remotefs && (retc=un_file.Serialize(local_path,XrdOssDIR|XrdOssEXC)) < 0)
       return retc;

// Check if this path is really a directory of a symbolic link elsewhere
//
    if (lstat(local_path, &statbuff)) retc = (errno == ENOENT ? 0 : -errno);
       else if ((statbuff.st_mode & S_IFMT) == S_IFLNK)
               retc = BreakLink(local_path, statbuff);
               else if ((statbuff.st_mode & S_IFMT) == S_IFDIR)
                       {if (remotefs) un_file.UnSerialize(0);
                        un_file.NoSerialize(local_path, XrdOssDIR);
                        if ((retc = rmdir(local_path))) retc = -errno;
                        DEBUG("dir rc=" <<retc <<" path=" <<local_path);
                        return retc;
                       } else doAdjust = 1;

// Delete the local copy and every valid suffix variation
//
   if (!retc)
      {if (unlink(local_path)) retc = -errno;
          else {i = strlen(local_path); fnp = &local_path[i];
                if (doAdjust && statbuff.st_size)
                   XrdOssCache::Adjust(statbuff.st_dev, -statbuff.st_size);
                if (haslf) for (i = 0; i < XrdOssPath::sfxMigL; i++)
                   {strcpy(fnp, XrdOssPath::Sfx[i]);
                    if (unlink(local_path))
                       if (errno == ENOENT) continue;
                          else retc2 = errno;
                       else retc2 = 0;
                    DEBUG("sfx retc=" <<retc2 <<' ' <<local_path);
                   }
               }
       DEBUG("lcl rc=" <<retc <<" path=" <<local_path);
      }

// If local copy effectively deleted. delete the remote copy if need be
//
   if (remotefs && !(Opts & XRDOSS_Online) 
   && (!retc || retc == -ENOENT) && RSSCmd)
      {if ((retc2 = MSS_Unlink(remote_path)) != -ENOENT) retc = retc2;
       DEBUG("rmt rc=" <<retc2 <<" path=" <<remote_path);
      }

// Deserialize the directory and return
//
   if (remotefs) un_file.UnSerialize(0);
   return retc;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                             B r e a k L i n k                              */
/******************************************************************************/

int XrdOssSys::BreakLink(const char *local_path, struct stat &statbuff)
{
    EPNAME("BreakLink")
    char *lP, lnkbuff[MAXPATHLEN+64];
    int lnklen, retc = 0;

// Read the contents of the link
//
    if ((lnklen = readlink(local_path, lnkbuff, sizeof(lnkbuff)-1)) < 0)
       return -errno;

// Return the actual stat information on the target (which might not exist
//
   lnkbuff[lnklen] = '\0';
   if (stat(lnkbuff, &statbuff)) statbuff.st_size = 0;
      else if (unlink(lnkbuff) && errno != ENOENT)
              {retc = -errno;
               OssEroute.Emsg("BreakLink",retc,"unlink symlink target",lnkbuff);
              } else {DEBUG("broke link " <<local_path <<"->" <<lnkbuff);}

// If this is a new-style cache, then we must also remove the pfn file.
// In any case, return the appropriate cache group.
//
   lP = lnkbuff+lnklen-1;
   if (*lP == XrdOssPath::xChar)
      {strcpy(lP+1, ".pfn"); unlink(lnkbuff);
       if (statbuff.st_size)
          {XrdOssPath::Trim2Base(lP);
           XrdOssCache::Adjust(lnkbuff, -statbuff.st_size);
          }
      } else if (statbuff.st_size)
                XrdOssCache::Adjust(statbuff.st_dev, -statbuff.st_size);

// All done
//
   return retc;
}
