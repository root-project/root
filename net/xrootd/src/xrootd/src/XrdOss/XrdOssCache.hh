#ifndef __XRDOSS_CACHE_H__
#define __XRDOSS_CACHE_H__
/******************************************************************************/
/*                                                                            */
/*                        X r d O s s C a c h e . h h                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <time.h>
#include <sys/stat.h>
#include "XrdOuc/XrdOucDLlist.hh"
#include "XrdSys/XrdSysError.hh"

/******************************************************************************/
/*              O S   D e p e n d e n t   D e f i n i t i o n s               */
/******************************************************************************/

#ifdef __solaris__
#include <sys/statvfs.h>
#define STATFS_t struct statvfs
#define FS_Stat(a,b) statvfs(a,b)
#define FS_BLKSZ f_frsize
#endif
#ifdef __linux__
#include <sys/vfs.h>
#define FS_Stat(a,b) statfs(a,b)
#define STATFS_t struct statfs
#define FS_BLKSZ f_bsize
#endif
#ifdef AIX
#include <sys/statfs.h>
#define STATFS_t struct statfs
#define FS_Stat(a,b) statfs(a,b)
#define FS_BLKSZ f_bsize
#endif
#ifdef __macos__
#include <sys/param.h>
#include <sys/mount.h>
#define STATFS_t struct statfs
#define FS_Stat(a,b) statfs(a,b)
#define FS_BLKSZ f_bsize
#endif

/******************************************************************************/
/*                    X r d O s s C a c h e _ F S D a t a                     */
/******************************************************************************/
  
// Flags values for FSData
//
#define XrdOssFSData_OFFLINE  0x0001
#define XrdOssFSData_ADJUSTED 0x0002
#define XrdOssFSData_REFRESH  0x0004

class XrdOssCache_FSData
{
public:

XrdOssCache_FSData *next;
long long           size;
long long           frsz;
dev_t               fsid;
const char         *path;
time_t              updt;
int                 stat;

       XrdOssCache_FSData(const char *, STATFS_t &, dev_t);
      ~XrdOssCache_FSData() {if (path) free((void *)path);}
};

/******************************************************************************/
/*                        X r d O s s C a c h e _ F S                         */
/******************************************************************************/

class XrdOssCache_Group;
  
class XrdOssCache_FS
{
public:

enum FSOpts {None = 0, isXA = 1};

XrdOssCache_FS     *next;
const   char       *group;
const   char       *path;
int                 plen;
FSOpts              opts;
        char        suffix[4];  // Corresponds to OssPath::sfxLen
XrdOssCache_FSData *fsdata;
XrdOssCache_Group  *fsgroup;

static long long    freeSpace(long long &Size, const char *path=0);

       XrdOssCache_FS(      int  &retc,
                      const char *fsg,
                      const char *fsp,
                      FSOpts      opt);
      ~XrdOssCache_FS() {if (group) free((void *)group);
                         if (path)  free((void *)path);
                        }
};

/******************************************************************************/
/*                     X r d O s s C a c h e _ G r o u p                      */
/******************************************************************************/
  
// Eventually we will have management information associated with cache groups
//
class XrdOssCache_Group
{
public:

XrdOssCache_Group *next;
char              *group;
XrdOssCache_FS    *curr;
long long          Usage;
long long          Quota;
int                GRPid;
static long long   PubQuota;

static XrdOssCache_Group *fsgroups;

       XrdOssCache_Group(const char *grp, XrdOssCache_FS *fsp=0) 
                        : next(0), group(strdup(grp)), curr(fsp), Usage(0),
                          Quota(-1), GRPid(-1) {}
      ~XrdOssCache_Group() {if (group) free((void *)group);}
};

// Suffixes to filenames that may exist in the cache
//
#define XRDOSS_SFX_LIST  (char *)".anew", (char *)".fail", (char *)".lock", (char *)".map", (char *)".stage"
#endif
