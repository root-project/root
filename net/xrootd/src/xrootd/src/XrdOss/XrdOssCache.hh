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
/*                       o o s s _ C a c h e _ L o c k                        */
/******************************************************************************/

class XrdOssCache_Lock
{
public:
void   Lock();
void UnLock();
     XrdOssCache_Lock();
    ~XrdOssCache_Lock();
private:
int locked;
};
  
/******************************************************************************/
/*                       X r d O s s C a c h e _ R e q                        */
/******************************************************************************/
  
// Flag values
//
#define XRDOSS_REQ_FAIL 0x0080
#define XRDOSS_REQ_ACTV 0x0001

struct XrdOssCache_Req
{
XrdOucDLlist<XrdOssCache_Req> fullList;
XrdOucDLlist<XrdOssCache_Req> pendList;

unsigned long               hash;         // Hash value for the path
const    char              *path;
unsigned long long          size;
int                         flags;
time_t                      sigtod;
int                         prty;

      XrdOssCache_Req(unsigned long xhash=0, const char *xpath=0)
          {hash  = xhash; fullList.setItem(this); pendList.setItem(this);
           if (xpath) path  = strdup(xpath);
              else path = 0;
           flags=0; sigtod=0; size=static_cast<long long>(2)<<31; prty=0;
          }
     ~XrdOssCache_Req() {if (path) free((void *)path);
                        fullList.Remove();
                        pendList.Remove();
                       }
};

/******************************************************************************/
/*                    X r d O s s C a c h e _ F S D a t a                     */
/******************************************************************************/
  
// Flags values for FSData
//
#define XrdOssFSData_OFFLINE  0x0001
#define XrdOssFSData_ADJUSTED 0x0002
#define XrdOssFSData_REFRESH  0x0004

struct XrdOssCache_FSData
{
XrdOssCache_FSData *next;
long long           size;
long long           frsz;
dev_t               fsid;
const char         *path;
time_t              updt;
int                 stat;

       XrdOssCache_FSData(const char *fsp, STATFS_t &fsbuff, dev_t fsID);
      ~XrdOssCache_FSData() {if (path) free((void *)path);}
};

/******************************************************************************/
/*                        X r d O s s C a c h e _ F S                         */
/******************************************************************************/
  
struct XrdOssCache_FS
{
XrdOssCache_FS     *next;
const   char       *group;
const   char       *path;
int                 plen;
XrdOssCache_FSData *fsdata;

       XrdOssCache_FS(      int  &retc, // Yucky historical output value
                      const char *fsg,
                      const char *fsp, 
                      const int inplace=0);
      ~XrdOssCache_FS() {if (group) free((void *)group);
                        if (path)  free((void *)path);
                       }
};

/******************************************************************************/
/*                     X r d O s s C a c h e _ G r o u p                      */
/******************************************************************************/
  
// Eventually we will have management information associated with cache groups
//
struct XrdOssCache_Group
{
XrdOssCache_Group  *next;
char              *group;
XrdOssCache_FS     *curr;

       XrdOssCache_Group(const char *grp, XrdOssCache_FS *fsp=0) 
                       {group = strdup(grp); curr = fsp; next = 0;}
      ~XrdOssCache_Group() {if (group) free((void *)group);}
};

// Suffixes to filenames that may exist in the cache
//
#define XRDOSS_SFX_LIST  (char *)".anew", (char *)".fail", (char *)".lock", (char *)".map", (char *)".stage"
#define XRDOSS_FAIL_FILE (char *)".fail"
#endif
