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
#include <utime.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssPath.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucPList.hh"

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

int XrdOssSys::Stat(const char *path, struct stat *buff, int opts)
{
    const int ro_Mode = ~(S_IWUSR | S_IWGRP | S_IWOTH);
    char actual_path[MAXPATHLEN+1], *local_path, *remote_path;
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

// Stat the file in the local filesystem first. If there. make sure the mode
// bits correspond to our reality and update access time if so requested.
//
   if (!stat(local_path, buff)) 
      {if (popts & XRDEXP_NOTRW) buff->st_mode &= ro_Mode;
       if (opts & XRDOSS_updtatm && (buff->st_mode & S_IFMT) == S_IFREG)
          {struct utimbuf times;
           times.actime  = time(0);
           times.modtime = buff->st_mtime;
           utime(local_path, &times);
          }
       return XrdOssOK;
      }

// The file may be offline in a mass storage system, check if this is possible
//
   if (!IsRemote(path) || opts & XRDOSS_resonly) return -errno;
   if (!RSSCmd) return (popts & XRDEXP_NOCHECK ? -ENOENT : -ENOMSG);

// Generate remote path
//
   if (rmt_N2N)
      if ((retc = rmt_N2N->lfn2rfn(path, actual_path, sizeof(actual_path))))
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
   int sVal, wVal, Util;
   long long fSpace, fSize;
   unsigned long long Opt;

// Get the values for this file system
//
   StatFS(path, Opt, fSize, fSpace);
   sVal = (Opt & XRDEXP_STAGE ? 1 : 0);
   wVal = (Opt & XRDEXP_NOTRW ? 0 : 1);

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

int XrdOssSys::StatFS(const char *path, unsigned long long &Opt,
                      long long &fSize, long long &fSpace)
{
// Establish the path options
//
   Opt = PathOpts(path);

// For in-place paths we just get the free space in that partition, otherwise
// get the maximum available in any partition.
//
   if ((Opt & XRDEXP_STAGE) || !(Opt & XRDEXP_NOTRW))
      if ((Opt & XRDEXP_INPLACE) || !XrdOssCache_Group::fsgroups)
         {char lcl_path[MAXPATHLEN+1];
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
   struct stat sbuff;
   XrdOssCache_Space   CSpace;
   char *cgrp, cgbuff[XrdOssSpace::minSNbsz];
   int retc;

// We provide psuedo support whould be not have a cache
//
   if (!XrdOssCache_Group::fsgroups)
      {unsigned long long Opt;
       long long fSpace, fSize;
       StatFS(path, Opt, fSize, fSpace);
       if (fSpace < 0) fSpace = 0;
       blen = snprintf(buff, blen, Resp, "public", fSize, fSpace, fSpace,
                                   fSize-fSpace, XrdOssCache_Group::PubQuota);
       return XrdOssOK;
      }

// Find the cache group. We provide psuedo support should we not have a cache
//
   if (!(cgrp = env.Get(OSS_CGROUP)))
      {if ((retc = getCname(path, &sbuff, cgbuff))) return retc;
          else cgrp = cgbuff;
      }

// Accumulate the stats and format the result
//
   blen = (XrdOssCache_FS::getSpace(CSpace, cgrp)
        ? snprintf(buff,blen,Resp,cgrp,CSpace.Total,CSpace.Free,CSpace.Maxfree,
                                       CSpace.Usage,CSpace.Quota)
        : snprintf(buff, blen, Resp, cgrp, 0LL, 0LL, 0LL, 0LL, -1LL));
   return XrdOssOK;
}

/******************************************************************************/
/*                                S t a t V S                                 */
/******************************************************************************/
  
/*
  Function: Return space information for space name "sname".

  Input:    sname       - The name of the same, null if all space wanted.
            sP          - pointer to XrdOssVSInfo to hold information.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
            Note that quota is zero when sname is null.
*/

int XrdOssSys::StatVS(XrdOssVSInfo *sP, const char *sname, int updt)
{
   XrdOssCache_Space   CSpace;

// Check if we should update the statistics
//
   if (updt) XrdOssCache::Scan(0);

// If no space name present or no spaces defined and the space is public then
// return information on all spaces.
//
   if (!sname || (!XrdOssCache_Group::fsgroups && !strcmp("public", sname)))
      {XrdOssCache::Mutex.Lock();
       sP->Total  = XrdOssCache::fsTotal;
       sP->Free   = XrdOssCache::fsTotFr;
       sP->LFree  = XrdOssCache::fsFree;
       sP->Large  = XrdOssCache::fsLarge;
       sP->Extents= XrdOssCache::fsCount;
       XrdOssCache::Mutex.UnLock();
       return XrdOssOK;
      }

// Get the space stats
//
   if (!(sP->Extents=XrdOssCache_FS::getSpace(CSpace,sname))) return -ENOENT;

// Return the result
//
   sP->Total = CSpace.Total;
   sP->Free  = CSpace.Free;
   sP->LFree = CSpace.Maxfree;
   sP->Large = CSpace.Largest;
   sP->Usage = CSpace.Usage;
   sP->Quota = CSpace.Quota;
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
   char cgbuff[XrdOssSpace::minSNbsz], fType;
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
/*                                S t a t X P                                 */
/******************************************************************************/

/*
  Function: Return export attributes for a path.

  Input:    path        - Is the path whose export attributes are wanted.
            attr        - reference to the are to receive the export attributes

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/

int XrdOssSys::StatXP(const char *path, unsigned long long &attr)
{

// Construct the processing options for this path
//
   attr = PathOpts(path);
   return XrdOssOK;
}
  
/******************************************************************************/
/*                              g e t C n a m e                               */
/******************************************************************************/
  
int XrdOssSys::getCname(const char *path, struct stat *sbuff, char *cgbuff)
{
   const char *thePath;
   char actual_path[MAXPATHLEN+1];
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
   if ((retc = stat(thePath, sbuff))) return retc;

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

/******************************************************************************/
/*                              g e t S t a t s                               */
/******************************************************************************/
  
int XrdOssSys::getStats(char *buff, int blen)
{
   static const char ptag1[] = "<paths>%d";
   static const char ptag2[] = "<stats id=\"%d\"><lp>\"%s\"</lp><rp>\"%s\"</rp>"
   "<tot>%lld</tot><free>%lld</free><ino>%lld</ino><ifr>%lld</ifr></stats>";
   static const char ptag3[] = "</paths>";

   static const int ptag1sz = sizeof(ptag1);
   static const int ptag2sz = sizeof(ptag2) + (16*4);
   static const int ptag3sz = sizeof(ptag3);

   static const char stag1[] = "<space>%d";
   static const char stag2[] = "<stats id=\"%d\"><name>%s</name>"
                "<tot>%lld</tot><free>%lld</free><maxf>%lld</maxf>"
                "<fsn>%d</fsn><usg>%lld</usg>";
   static const char stagq[] = "<qta>%lld</qta>";
   static const char stags[] = "</stats>";
   static const char stag3[] = "</space>";

   static const int stag1sz = sizeof(stag1);
   static const int stag2sz = sizeof(stag2) + XrdOssSpace::maxSNlen + (16*5);
   static const int stagqsz = sizeof(stagq) + 16;
   static const int stagssz = sizeof(stags);
   static const int stag3sz = sizeof(stag3);

   static const int stagsz  = ptag1sz + ptag2sz + ptag3sz + 1024 +
                            + stag1sz + stag2sz + stag3sz
                            + stagqsz + stagssz;

   XrdOssCache_Group  *fsg = XrdOssCache_Group::fsgroups;
   XrdOssCache_Space   CSpace;
   OssDPath           *dpP = DPList;
   char *bp = buff;
   int dpNum = 0, spNum = 0, n, flen;

// If no buffer spupplied, return how much data we will generate. We also
// do one-time initialization here.
//
   if (!buff) return ptag1sz + (ptag2sz * numDP) + stag3sz + lenDP
                   + stag1sz + (stag2sz * numCG) + stag3sz
                   + stagqsz + stagssz;

// Make sure we have enough space for one entry
//
   if (blen <= stagsz) return 0;

// Output first header (we know we have one path, at least)
//
   flen = sprintf(bp, ptag1, numDP); bp += flen; blen -= flen;

// Output individual entries
//
   while(dpP && blen > 0)
        {XrdOssCache_FS::freeSpace(CSpace, dpP->Path2);
         flen = snprintf(bp, blen, ptag2, dpNum, dpP->Path1, dpP->Path2,
                                   CSpace.Total>>10, CSpace.Free>>10,
                                   CSpace.Inodes,    CSpace.Inleft);
         dpP = dpP->Next; bp += flen; blen -= flen; dpNum++;
        }

// Output closing tag
//
   if (blen <= ptag3sz) return 0;
   strcpy(bp, ptag3); bp += (ptag3sz-1); blen -= (ptag3sz-1);
   dpNum = bp - buff;

// Output header
//
   if (blen <= stag1sz) return (blen < 0 ? 0 : dpNum);
   flen = snprintf(bp, blen, stag1, numCG); bp += flen; blen -= flen;
   if (blen <= stag1sz) return dpNum;

// Generate info for each path
//
   while(fsg && blen > 0)
        {n = XrdOssCache_FS::getSpace(CSpace, fsg);
         flen = snprintf(bp, blen, stag2, spNum, fsg->group, CSpace.Total>>10,
                CSpace.Free>>10, CSpace.Maxfree>>10, n, CSpace.Usage>>10);
         bp += flen; blen -= flen; spNum++;
         if (CSpace.Quota >= 0 && blen > stagqsz)
            {flen = sprintf(bp, stagq, CSpace.Quota); bp += flen; blen -= flen;}
         if (blen < stagssz) return dpNum;
         strcpy(bp, stags); bp += (stagssz-1); blen -= (stagssz-1);
         fsg = fsg->next;
        }

// Insert trailer
//
   if (blen >= stag3sz) {strcpy(bp, stag3); bp += (stag3sz-1);}
      else return dpNum;

// All done
//
   return bp - buff;
}
