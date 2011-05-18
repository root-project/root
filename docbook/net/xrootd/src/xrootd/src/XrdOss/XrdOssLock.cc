/******************************************************************************/
/*                                                                            */
/*                         X r d O s s L o c k . c c                          */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

const char *XrdOssLockCVSID = "$Id$";

/* The XrdOssSerialize() and XrdOssUnSerialize() routines are responsible for
   serializing access to directories and files. The current implementaion
   uses flock with a hiearchical system of locking. The defined protocol is:

   Stage in a file:
   1) Exclusively flock the DIR_LOCK file in the target directory.
   2) Create the shadow lock file.
   3) Exclusively flock the shadow lock file for the file to be staged in.
   4) Unlock the directory.
   5) Atomically copy the file to the local file system.
   6) Set the shadow lock file mtime to be the same as the base file's mtime.
   7) Unlock the lock file.

   Open a file:
   1) Exclusively flock the DIR_LOCK file in the target directory.
   2) Open the file.
   3) Unlock the directory.
   4) Do whatever with it.

   Stage out file:
   1) Exclusively flock the DIR_LOCK file in the target directory.
   2) Exclusively flock the shadow lock file for the file to be staged out.
   3) Unlock the directory.
   4) Copy the file to the target storage system.
   5) Check whether the file has been modified during the copy. If so, the
      copy is invalidated and the file needs to be copied again.
   6) Set the shadow lock file mtime to be the same as the base file's mtime.
   7) Unlock the lock file.

   Purge a file:
   1) Exclusively flock the DIR_LOCK file in the target directory.
   2) Attempt a share lock on the shadow lock file for the file to be purged.
      If someone else has a lock, skip to step 6.
   3) Check if the base file is open (e.g., via fuser). If it is, skip
      to step 5.
   4) Check if the lock file mtime >= base file mtime. If it is, purge the file
      as well as the corresponding lock file.
   5) Unlock the lock file.
   6) Unlock the directory lock.


   These routines are thread-safe if compiled with:
   AIX: -D_THREAD_SAFE
   SUN: -D_REENTRANT
*/

/******************************************************************************/
/*                               i n c l u d e s                              */
/******************************************************************************/
  
#include <unistd.h>
#include <stdio.h>
#include <sys/file.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <strings.h>
#include <utime.h>

#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssLock.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*           G l o b a l   E r r o r   R o u t i n g   O b j e c t            */
/******************************************************************************/
  
extern XrdSysError OssEroute;

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define XrdOssLKFNAME  "DIR_LOCK"
#define XrdOssLKSUFFIX ".lock"
#define XrdOssLKTRIES  300
#define XrdOssLKWAIT   1

/******************************************************************************/
/*                         o o s s _ S e r i a l i z e                        */
/******************************************************************************/

/*In:  fn      - The path to be locked.
       lkwant  - Locking options:
                 XrdOssDIR      - Lock the corresponding directory.
                 XrdOssFILE     - Lock the target file.
                 XrdOssNOWAIT   - Do not block.
                 XrdOssEXC      - Exclusive lock (the default).
                 XrdOssSHR      - Shared lock.
                 XrdOssRETIME   - Adjust time for relativistic creation effects

   Out: XrdOssOK upon success; -errno,otherwise.
*/

int XrdOssLock::Serialize(const char *fn, int lkwant)
{
    char lkbuff[MAXPATHLEN+sizeof(XrdOssLKFNAME)];
    int rc;

// Check if this object is already in use
//
   if (lkfd >= 0) 
      return OssEroute.Emsg("Serialize",-XRDOSS_E8014,"lock",lkbuff);

// Create the lock file name that we will lock as requested.
//
    if ((rc = Build_LKFN(lkbuff, sizeof(lkbuff), fn, lkwant))) return rc;

// Open the file in write mode (create it if not there).
//
    do { lkfd = open(lkbuff, O_RDWR|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR);}
        while( lkfd < 0 && errno == EINTR);
    if ( lkfd < 0) 
       {if (ENOENT != (rc = errno))
           OssEroute.Emsg("Serialize",rc,"serially open",lkbuff);
        return -rc;
       }
    fcntl(lkfd, F_SETFD, FD_CLOEXEC);

// If we should adjust time, do so now
//
   if (lkwant & XrdOssRETIME)
      {struct stat    buf;
       struct utimbuf times;
       if (!(rc = stat(fn, &buf)))
          {times.actime =  buf.st_atime;
           times.modtime = buf.st_mtime-63;
           rc = utime(lkbuff, (const struct utimbuf *)&times);
          }
       if (rc) {rc = errno; close(lkfd); lkfd = -1;
                return OssEroute.Emsg("Serialize",rc,"retime",lkbuff);
               }
      }

// Now lock the file and return the file descriptor.
//
    if ((rc = XLock(lkfd, lkwant)))
       {char *mp;
        close(lkfd); lkfd = -1;
        if (rc == EWOULDBLOCK) return -EWOULDBLOCK;
        if (lkwant & XrdOssRETIME)
           mp = (lkwant&XrdOssSHR ? (char *)"rt shr lk":(char *)"rt exc lk");
           else mp = (lkwant & XrdOssSHR ? (char *)"shr lk":(char *)"exc lk");
        return OssEroute.Emsg("Serialize", rc, mp, lkbuff);
        return -XRDOSS_E8015;
       }
    return XrdOssOK;
}

/******************************************************************************/
/*                        o o s s _ N o S e r i a l i z e                     */
/******************************************************************************/

/*In:  fn      - The filename whose lockfile is to be deleted.
       ftype   - Lock type (one must be specified):
                 XrdOssDIR      - Directory lock.
                 XrdOssFILE     - File lock.

   Out: Upon success, zero is returned.
        Otherwise, a negative error code is returned corresponding to -errno.
*/

int XrdOssLock::NoSerialize(const char *fn, int ftype)
{
    char lkbuff[MAXPATHLEN+sizeof(XrdOssLKFNAME)];
    int rc;

// Verify that a lock filetype has been specified.
//
    if (!(ftype & (XrdOssDIR | XrdOssFILE)))
       return OssEroute.Emsg("NoSerialize", -XRDOSS_E8016,
                               "unserialize fname", (char *)fn);

// Create the lock file name that we will lock as requested.
//
    if ((rc = Build_LKFN(lkbuff, sizeof(lkbuff), fn, ftype))) return rc;

// Unlink the file.
//
    if (unlink(lkbuff))
       {rc = errno;
        if (rc != ENOENT) 
           return OssEroute.Emsg("NoSerialize", -rc,
                                   "unserialize lkfname", (char *)fn);
       }
    return XrdOssOK;
}

/******************************************************************************/
/*                   o o s s _ R e S e r i a l i z e                          */
/******************************************************************************/

/*In:  oldname - The old name of the base file being renamed.
       newname - The new name of the base file.

  Out: If the corresponding lock file is rename is successfully renamed,
       zero is returned. Otherwise -errno is returned.

 Note: The correspodning directory must have been locked by the caller!
*/

int XrdOssLock::ReSerialize(const char *oldname, const char *newname)
{
    int rc = 0;
    char Path_Old[MAXPATHLEN+1];
    char Path_New[MAXPATHLEN+1];

// Build old and new lock file names
//
   if ((rc = Build_LKFN(Path_Old, sizeof(Path_Old), oldname, XrdOssFILE)))
      return rc;
   if ((rc = Build_LKFN(Path_New, sizeof(Path_New), newname, XrdOssFILE)))
      return rc;

// Rename the lock file.
//
   if (rename(Path_Old, Path_New))
      {rc = errno;
       if (rc != ENOENT) OssEroute.Emsg("ReSerialize",rc,"reserialize",Path_Old);
          else rc = 0;
      }
   return -rc;
}

/******************************************************************************/
/*                   o o s s _ U n S e r i a l i z e                          */
/******************************************************************************/

/*In:  opts    - Unlocking options:
                 XrdOssLEAVE    - leave the underlying filehandle open.
                 XrdOssREGRADE  - Don't release the lock. Instead do an upgrade
                                 or a downgrade (default is to release lock).
                 XrdOssRETRY    - release the lock and pause (if !XrdOssREGRADE),
                                 then try to obtain the lock again with
                                 XrdOssSerialize() options.
*/

int XrdOssLock::UnSerialize(int opts)
{
    int maxtry = XrdOssLKTRIES;
    int xopts, rc, dosleep = 1;
    const struct timespec naptime = {XrdOssLKWAIT, 0};

// Check if we havenything reallly locked
//
   if (lkfd < 0) 
      return OssEroute.Emsg("UnSerialize",-XRDOSS_E8017,"unserialize lock");

// Release the lock if we need to.
//
   if (!(opts & XrdOssREGRADE)) XLock(lkfd, 0);
      else dosleep = 0;

// Based on execution option, perform the required action.
//
    xopts = opts & (XrdOssLEAVE | XrdOssRETRY);
    switch(xopts)
         {case XrdOssLEAVE: break;
          case XrdOssRETRY: do {if (dosleep) nanosleep(&naptime, 0);
                               if (! (rc = XLock(lkfd, opts)) ) break;
                               dosleep = 1;
                              } while( rc == EWOULDBLOCK && 
                                      !(opts & XrdOssNOWAIT) && maxtry--);
                           return -rc;
          default:         close(lkfd); lkfd = -1;
                           break;
         }
    return XrdOssOK;
}

/******************************************************************************/
/*                            B u i l d _ L K F N                             */
/******************************************************************************/

int XrdOssLock::Build_LKFN(char *buff, int blen, const char *fn, int ftype)
{  int i;

// Verify that input filename is not too large.
//
   i = strlen(fn);
   if (i + (ftype & XrdOssFILE ? (int)sizeof(XrdOssLKSUFFIX) 
                               : (int)sizeof(XrdOssLKFNAME)+1) > blen)
      return OssEroute.Emsg("Build_LKFN", -ENAMETOOLONG,
                              "generate lkfname", (char *)fn);

// Create the lock file name that we will lock in exclusive mode.
//
   strcpy(buff, fn);
   if (ftype & XrdOssFILE) strcat(buff, XrdOssLKSUFFIX);
      else {
            for (i = i-1; i >= 0 && buff[i] != '/'; i--){}
            if (i <= 0) {strcpy(buff, "./"); i = 1;}
            strcpy(&buff[i+1], XrdOssLKFNAME);
           }

// All done.
//
   return XrdOssOK;
}

/******************************************************************************/
/*                                X L o c k                                   */
/******************************************************************************/

int XrdOssLock::XLock(int lkFD, int opts)
{
    FLOCK_t lock_args;

// Make sure we have a lock outstanding
//
    if (lkFD < 0) return XrdOssOK;

// Establish locking options
//
    bzero(&lock_args, sizeof(lock_args));
    if (opts & XrdOssSHR) lock_args.l_type = F_RDLCK;
       else if (opts & XrdOssEXC) lock_args.l_type = F_WRLCK;
               else lock_args.l_type = F_UNLCK;

// Perform action.
//
    if (fcntl(lkFD, (opts & XrdOssNOWAIT ? F_SETLK : F_SETLKW),
                    &lock_args)) return errno;
    return XrdOssOK;
}
