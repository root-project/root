/******************************************************************************/
/*                                                                            */
/*                         X r d O s s S t a t . c c                          */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOssStatCVSID = "$Id$";

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
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucName2Name.hh"

/******************************************************************************/
/*                                 s t a t                                    */
/******************************************************************************/

/*
  Function: Determine if file 'path' actually exists.

  Input:    path        - Is the fully qualified name of the file to be tested.
            buff        - pointer to a 'stat' structure to hold the attributes
                          of the file.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/

int XrdOssSys::Stat(const char *path, struct stat *buff, int resonly)
{
    const int ro_Mode = ~(S_IWUSR | S_IWGRP | S_IWOTH);
    char actual_path[XrdOssMAX_PATH_LEN+1], *local_path, *remote_path;
    unsigned long long popts;
    int retc;

// Construct the processing options for this path
//
   popts = PathOpts(path);

// Generate local path
//
   if (lcl_N2N)
      if ((retc = lcl_N2N->lfn2pfn(path, actual_path, sizeof(actual_path)))) 
         return retc;
         else local_path = actual_path;
      else local_path = (char *)path;

// Stat the file in the local filesystem first.
//
   if (!stat(local_path, buff)) 
      {if (popts & XRDEXP_NOTRW) buff->st_mode &= ro_Mode;
       return XrdOssOK;
      }
   if (!IsRemote(path)) return -errno;
   if (resonly) return -ENOMSG;

// Generate remote path
//
   if (rmt_N2N)
      if ((retc = rmt_N2N->lfn2pfn(path, actual_path, sizeof(actual_path))))
         return retc;
         else remote_path = actual_path;
      else remote_path = (char *)path;

// Now stat the file in the remote system (it doesn't exist locally)
//
   if ((retc = MSS_Stat(remote_path, buff))) return retc;
   if (popts & XRDEXP_NOTRW) buff->st_mode &= ro_Mode;
   buff->st_mode |= S_IFBLK;
   return XrdOssOK;
}

/******************************************************************************/
/*                                S t a t F S                                 */
/******************************************************************************/

/*
  Function: Return free space information based on a path

  Input:    path        - Is the fully qualified name of the file to be tested.
            buff        - pointer to a buffer to hold the information.
            blen        - the length of the buffer

  Output:   Returns XrdOssOK upon success and -errno upon failure.
            blen is updated with the actual length of the buff data.
*/

int XrdOssSys::StatFS(const char *path, char *buff, int &blen)
{
   int Opt, sVal, wVal, Util;
   long long fSpace, fSize;

// Get the values for this file system
//
   StatFS(path, Opt, fSize, fSpace);
   sVal = (Opt & XRDEXP_REMOTE ? 1 : 0);
   wVal = (Opt & XRDEXP_NOTRW  ? 0 : 1);

// Size the value to fit in an int
//
   if (fSpace <= 0) {fSize = fSpace = 0; Util = 0;}
      else {Util = (fSize ? (fSize - fSpace)*100LL/fSize : 0);
            fSpace = fSpace >> 20LL;
            if ((fSpace >> 31LL)) fSpace = 0x7fffffff;
           }

// Return the result
//
   blen = snprintf(buff, blen, "%d %lld %d %d %lld %d",
                               wVal, (wVal ? fSpace : 0LL), (wVal ? Util : 0),
                               sVal, (sVal ? fSpace : 0LL), (sVal ? Util : 0));
   return XrdOssOK;
}

/******************************************************************************/

/*
  Function: Return free space information based on a path

  Input:    path        - Is the fully qualified name of the file to be tested.
            opt         - Options associated with the path
            fSize       - total bytes in the filesystem.
            fSpace      - total free bytes in the filesystem. It is set to
                          -1 if the path is not convertable.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/

int XrdOssSys::StatFS(const char *path, 
                      int &Opt, long long &fSize, long long &fSpace)
{

// For in-place paths we just get the free space in that partition, otherwise
// get the maximum available in any partition.
//
   if ((Opt & XRDEXP_REMOTE) || !(Opt & XRDEXP_NOTRW))
      if ((Opt & XRDEXP_INPLACE) || !XrdOssCache_Group::fsgroups)
         {char lcl_path[XrdOssMAX_PATH_LEN+1];
          if (lcl_N2N)
             if (lcl_N2N->lfn2pfn(path, lcl_path, sizeof(lcl_path)))
                fSpace = -1;
                else fSpace = XrdOssCache_FS::freeSpace(fSize, lcl_path);
             else    fSpace = XrdOssCache_FS::freeSpace(fSize, path);
         } else     {fSpace = XrdOssCache_FS::freeSpace(fSize);}
      else          {fSpace = 0;      fSize = 0;}
   return XrdOssOK;
}

/******************************************************************************/
/*                                S t a t L S                                 */
/******************************************************************************/

/*
  Function: Return free space information based on a cahe group name.

  Input:    Env         - Is the environment for cgi info.
            path        - Is the path name.
            buff        - pointer to a buffer to hold the information.
            blen        - the length of the buffer

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/

int XrdOssSys::StatLS(XrdOucEnv &env, const char *path, char *buff, int &blen)
{
   static const char *Resp="oss.cgroup=%s&oss.space=%lld&oss.free=%lld"
                           "&oss.maxf=%lld&oss.used=%lld&oss.quota=%lld";
   long long Tspace, Fspace, Mspace, Uspace, Quota;
   struct stat sbuff;
   XrdOssCache_Group  *fsg = XrdOssCache_Group::fsgroups;
   XrdOssCache_FS     *fsp;
   XrdOssCache_FSData *fsd;
   char *cgrp, cgbuff[64];
   int retc;

// We provide psuedo support whould be not have a cache
//
   if (!fsg)
      {int Opt;
       long long fSpace, fSize;
       StatFS(path, Opt, fSize, fSpace);
       if (fSpace < 0) fSpace = 0;
       blen = snprintf(buff, blen, Resp, "public", fSize, fSpace, fSpace,
                                   fSize-fSpace, XrdOssCache_Group::PubQuota);
       return XrdOssOK;
      }

// Find the cache group. We provide pshuedo support whould be not have a cache
//
   if (!(cgrp = env.Get(OSS_CGROUP)))
      {if ((retc = getCname(path, &sbuff, cgbuff))) return retc;
          else cgrp = cgbuff;
      }

// Try to find the cache group. If there is no cache and
//
   while(fsg && strcmp(cgrp, fsg->group)) fsg = fsg->next;
   if (!fsg)
      {blen = snprintf(buff, blen, Resp, cgrp, 0LL, 0LL, 0LL, 0LL, -1LL);
       return XrdOssOK;
      }

// Prepare to accumulate the stats
//
   Tspace = Fspace = Mspace = 0;
   CacheContext.Lock();
   Uspace = fsg->Usage; Quota = fsg->Quota;
   if ((fsp = fsfirst)) do
      {if (fsp->fsgroup == fsg)
          {fsd = fsp->fsdata;
           Tspace += fsd->size;    Fspace += fsd->frsz;
           if (fsd->frsz > Mspace) Mspace  = fsd->frsz;
          }
       fsp = fsp->next;
      } while(fsp != fsfirst);
   CacheContext.UnLock();

// Format the result
//
   blen = snprintf(buff,blen,Resp,cgrp,Tspace,Fspace,Mspace,Uspace,Quota);
   return XrdOssOK;
}

/******************************************************************************/
/*                                S t a t X A                                 */
/******************************************************************************/
  
/*
  Function: Return extended attributes for "path".

  Input:    path        - Is the fully qualified name of the target file.
            buff        - pointer to a buffer to hold the information.
            blen        - the length of the buffer

  Output:   Returns XrdOssOK upon success and -errno upon failure.
            blen is updated with the actual length of the buff data.
*/

int XrdOssSys::StatXA(const char *path, char *buff, int &blen)
{
   struct stat sbuff;
   char cgbuff[64], fType;
   long long Size, Mtime, Ctime, Atime;
   int retc;

// Get the cache group and stat info for the file
//
   if ((retc = getCname(path, &sbuff, cgbuff))) return retc;
        if (S_ISREG(sbuff.st_mode)) fType = 'f';
   else if (S_ISDIR(sbuff.st_mode)) fType = 'd';
   else                             fType = 'o';

// Format the result
//
   Size = sbuff.st_size;
   Mtime = sbuff.st_mtime; Ctime = sbuff.st_ctime; Atime = sbuff.st_atime;
   blen = snprintf(buff, blen, 
          "oss.cgroup=%s&oss.type=%c&oss.used=%lld&oss.mt=%lld"
          "&oss.ct=%lld&oss.at=%lld&oss.u=*&oss.g=*&oss.fs=%c",
          cgbuff, fType, Size, Mtime, Ctime, Atime,
          (sbuff.st_mode & S_IWUSR ? 'w':'r'));
   return XrdOssOK;
}

/******************************************************************************/
/*                              g e t C n a m e                               */
/******************************************************************************/
  
int XrdOssSys::getCname(const char *path, struct stat *sbuff, char *cgbuff)
{
   const char *thePath;
   char actual_path[XrdOssMAX_PATH_LEN+1];
   int retc;

// Get the pfn for this path
//
   if (lcl_N2N)
      if ((retc = lcl_N2N->lfn2pfn(path, actual_path, sizeof(actual_path))))
         return retc;
         else thePath = actual_path;
      else thePath = path;

// Get regular stat informtion for this file
//
   if ((retc = Stat(thePath, sbuff))) return retc;

// Now determine if we should get the cache group name. There is none
// for offline files and it's always public for directories.
//
   if (S_ISDIR(sbuff->st_mode))          strcpy(cgbuff, "public");
      else if (sbuff->st_mode & S_IFBLK) strcpy(cgbuff, "*");
              else XrdOssPath::getCname(thePath, cgbuff);

// All done
//
   return 0;
}
