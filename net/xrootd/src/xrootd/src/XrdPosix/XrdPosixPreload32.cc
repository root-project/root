/******************************************************************************/
/*                                                                            */
/*                  X r d P o s i x P r e l o a d 3 2 . c c                   */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdPosixPreload32CVSID = "$Id$";

#if !defined(SUNX86) && defined(__LP64__) && !defined(_LP64)
#undef  _LARGEFILE_SOURCE
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#endif
#define XRDPOSIXPRELOAD32

#include <errno.h>
#include <dirent.h>
#include <stdio.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__macos__) || defined(__FreeBSD__)
#include <sys/param.h>
#include <sys/mount.h>
#else
#include <sys/statfs.h>
#endif

#include "XrdPosix/XrdPosixExtern.hh"
#include "XrdPosix/XrdPosixLinkage.hh"
#include "XrdPosix/XrdPosixOsDep.hh"
#include "XrdPosix/XrdPosixXrootd.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
 
/******************************************************************************/
/*                   G l o b a l   D e c l a r a t i o n s                    */
/******************************************************************************/
  
extern XrdPosixLinkage Xunix;
 
/******************************************************************************/
/*               6 4 - t o 3 2   B i t   C o n v e r s i o n s                */
/******************************************************************************/
/******************************************************************************/
/*                   X r d P o s i x _ C o p y D i r e n t                    */
/******************************************************************************/
  
// Macos is a curious beast. It is not an LP64 platform but offsets are
// defined as 64 bits anyway. So, the dirent structure is 64-bit conformable
// making CopyDirent() superfluous. In Solaris x86 there are no 32 bit interfaces.
//
#if !defined(__LP64__) && !defined(_LP64)
#if !defined( __macos__) && !defined(SUNX86) && !defined(__FreeBSD__)
int XrdPosix_CopyDirent(struct dirent *dent, struct dirent64 *dent64)
{
  const unsigned long long LLMask = 0xffffffff00000000LL;
  int isdiff = (dent->d_name-(char *)dent) != (dent64->d_name-(char *)dent64);

  if (isdiff  && ((dent64->d_ino & LLMask) || (dent64->d_off & LLMask)))
     {errno = EOVERFLOW; return EOVERFLOW;}

  if (isdiff || (void *)dent != (void *)dent64)
     {dent->d_ino    = dent64->d_ino;
      dent->d_off    = dent64->d_off;
      dent->d_reclen = dent64->d_reclen;
      strcpy(dent->d_name, dent64->d_name);
     }
  return 0;
}
#endif
#endif

/******************************************************************************/
/*                     X r d P o s i x _ C o p y S t a t                      */
/******************************************************************************/
  
// Macos is a curious beast. It is not an LP64 platform but stat sizes are
// defined as 64 bits anyway. So, the stat structure is 64-bit conformable
// making CopyStat() seemingly superfluous. However, starting in Darwin 10.5
// stat and stat64 are defined separately making it necessary to use CopyStat().
// In Solaris x86 there are no 32 bit interfaces.
//
#if !defined(__LP64__) && !defined(_LP64)
#if !defined(SUNX86) && !defined(__FreeBSD__)
int XrdPosix_CopyStat(struct stat *buf, struct stat64 &buf64)
{
  const unsigned long long LLMask = 0xffffffff00000000LL;
  const      int  INTMax = 0x7fffffff;

  if (buf64.st_size   & LLMask)
     if (buf64.st_mode & S_IFREG || buf64.st_mode & S_IFDIR)
        {errno = EOVERFLOW; return -1;}
        else buf->st_size   = INTMax;
     else buf->st_size =  buf64.st_size;  /* 64: File size in bytes */

      buf->st_ino   = buf64.st_ino    & LLMask ? INTMax : buf64.st_ino;
      buf->st_blocks= buf64.st_blocks & LLMask ? INTMax : buf64.st_blocks;
      buf->st_mode  = buf64.st_mode;     /*     File mode (see mknod(2)) */
      buf->st_dev   = buf64.st_dev;
      buf->st_rdev  = buf64.st_rdev;     /*     ID of device */
      buf->st_nlink = buf64.st_nlink;    /*     Number of links */
      buf->st_uid   = buf64.st_uid;      /*     User ID of the file's owner */
      buf->st_gid   = buf64.st_gid;      /*     Group ID of the file's group */
      buf->st_atime = buf64.st_atime;    /*     Time of last access */
      buf->st_mtime = buf64.st_mtime;    /*     Time of last data modification */
      buf->st_ctime = buf64.st_ctime;    /*     Time of last file status change */
      buf->st_blksize=buf64.st_blksize;  /*     Preferred I/O block size */
  return 0;
}
#endif
#endif

/******************************************************************************/
/*                                 c r e a t                                  */
/******************************************************************************/
  
#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
int     creat(const char *path, mode_t mode)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Open(path, O_WRONLY | O_CREAT | O_TRUNC, mode);
}
}
#endif

/******************************************************************************/
/*                                 f c n t l                                  */
/******************************************************************************/
  
extern "C"
{
int     fcntl(int fd, int cmd, ...)
{
   static int Init = Xunix.Init(&Init);
   va_list ap;
   void *theArg;

   if (XrdPosixXrootd::myFD(fd)) return 0;
   va_start(ap, cmd);
   theArg = va_arg(ap, void *);
   va_end(ap);
   return Xunix.Fcntl(fd, cmd, theArg);
}
}
  
/******************************************************************************/
/*                                 f o p e n                                  */
/******************************************************************************/
/*
extern "C"
{
FILE  *fopen(const char *path, const char *mode)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fopen(path, mode);
}
}
*/

  
/******************************************************************************/
/*                                f s e e k o                                 */
/******************************************************************************/

#ifndef SUNX86
extern "C"
{
int fseeko(FILE *stream, off_t offset, int whence)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fseeko(stream, offset, whence);
}
}
#endif

/******************************************************************************/
/*                                 f s t a t                                  */
/******************************************************************************/

#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
#if defined __linux__ && __GNUC__ && __GNUC__ >= 2
int  __fxstat(int ver, int fildes, struct stat *buf)
#elif defined(__solaris__) && defined(__i386)
int   _fxstat(int ver, int fildes, struct stat *buf)
#else
int     fstat(         int fildes, struct stat *buf)
#endif
{
   static int Init = Xunix.Init(&Init);

#ifdef __linux__
   if (!XrdPosixXrootd::myFD(fildes)) return Xunix.Fstat(ver, fildes, buf);
#else
   if (!XrdPosixXrootd::myFD(fildes)) return Xunix.Fstat(     fildes, buf);
#endif

#if defined(__LP64__) || defined(_LP64)
   return    XrdPosix_Fstat(fildes,                 buf  );
#else
   int rc;
   struct stat64 buf64;
   if ((rc = XrdPosix_Fstat(fildes, (struct stat *)&buf64))) return rc;
   return XrdPosix_CopyStat(buf, buf64);
#endif
}
}
#endif

  
/******************************************************************************/
/*                                f t e l l o                                 */
/******************************************************************************/

#ifndef SUNX86
extern "C"
{
off_t ftello(FILE *stream)
{
   static int Init = Xunix.Init(&Init);

   return static_cast<off_t>(XrdPosix_Ftello(stream));
}
}
#endif

/******************************************************************************/
/*                             f t r u n c a t e                              */
/******************************************************************************/
  
#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
int ftruncate(int fildes, off_t offset)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Ftruncate(fildes, offset);
}
}
#endif

/******************************************************************************/
/*                                 l s e e k                                  */
/******************************************************************************/
  
#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
off_t   lseek(int fildes, off_t offset, int whence)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Lseek(fildes, offset, whence);
}
}
#endif

/******************************************************************************/
/*                                 l s t a t                                  */
/******************************************************************************/

#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
#if defined __GNUC__ && __GNUC__ >= 2 && defined(__linux__)
int     __lxstat(int ver, const char *path, struct stat *buf)
#elif defined(__solaris__) && defined(__i386)
int      _lxstat(int ver, const char *path, struct stat *buf)
#else
int        lstat(         const char *path, struct stat *buf)
#endif
{
   static int Init = Xunix.Init(&Init);

   if (!XrdPosix_isMyPath(path))
#ifdef __linux__
      return Xunix.Lstat(ver, path, buf);
#else
      return Xunix.Lstat(     path, buf);
#endif

#if defined(__LP64__) || defined(_LP64)
   return    XrdPosix_Lstat(path,                 buf  );
#else
   struct stat64 buf64;
   int rc;
   if ((rc = XrdPosix_Lstat(path, (struct stat *)&buf64))) return rc;
   return XrdPosix_CopyStat(buf, buf64);
#endif
}
}
#endif

/******************************************************************************/
/*                                  o p e n                                   */
/******************************************************************************/
  
#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
int     open(const char *path, int oflag, ...)
{
   static int Init = Xunix.Init(&Init);
   va_list ap;
   int mode;

   va_start(ap, oflag);
   mode = va_arg(ap, int);
   va_end(ap);
   return XrdPosix_Open(path, oflag, mode);
}
}
#endif

/******************************************************************************/
/*                                 p r e a d                                  */
/******************************************************************************/
  
#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
ssize_t pread(int fildes, void *buf, size_t nbyte, off_t offset)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Pread(fildes, buf, nbyte, offset);
}
}
#endif

/******************************************************************************/
/*                               r e a d d i r                                */
/******************************************************************************/

#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
struct dirent* readdir(DIR *dirp)
{
   static int Init = Xunix.Init(&Init);
   struct dirent64 *dp64;

   if (!(dp64 = XrdPosix_Readdir64(dirp))) return 0;

#if !defined(__macos__) && !defined(_LP64) && !defined(__LP64__)
   if (XrdPosix_CopyDirent((struct dirent *)dp64, dp64)) return 0;
#endif

   return (struct dirent *)dp64;
}
}
#endif

/******************************************************************************/
/*                             r e a d d i r _ r                              */
/******************************************************************************/
  
#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
int     readdir_r(DIR *dirp, struct dirent *entry, struct dirent **result)
{
   static int Init = Xunix.Init(&Init);
#if defined(__macos__) || defined(__LP64__) || defined(_LP64)
   return XrdPosix_Readdir_r(dirp, entry, result);
#else
   char buff[sizeof(struct dirent64) + 2048];
   struct dirent64 *dp64 = (struct dirent64 *)buff;
   struct dirent64 *mydirent;
   int rc;

   if ((rc = XrdPosix_Readdir64_r(dirp, dp64, &mydirent)))
      return rc;

   if (!mydirent) {*result = 0; return 0;}

   if ((rc = XrdPosix_CopyDirent(entry, dp64))) return rc;

   *result = entry;
   return 0;
#endif
}
}
#endif

/******************************************************************************/
/*                                p w r i t e                                 */
/******************************************************************************/
  
#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
ssize_t pwrite(int fildes, const void *buf, size_t nbyte, off_t offset)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Pwrite(fildes, buf, nbyte, offset);
}
}
#endif

/******************************************************************************/
/*                                  s t a t                                   */
/******************************************************************************/

#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
#if defined __GNUC__ && __GNUC__ >= 2
int     __xstat(int ver, const char *path, struct stat *buf)
#elif defined(__solaris__) && defined(__i386)
int      _xstat(int ver, const char *path, struct stat *buf)
#else
int        stat(         const char *path, struct stat *buf)
#endif
{
   static int Init = Xunix.Init(&Init);

   if (!XrdPosix_isMyPath(path))
#ifdef __linux__
      return Xunix.Stat(ver, path, buf);
#else
      return Xunix.Stat(     path, buf);
#endif

#if defined(__LP64__) || defined(_LP64)
   return    XrdPosix_Stat(path,                 buf  );
#else
   struct stat64 buf64;
   int rc;
   if ((rc = XrdPosix_Stat(path, (struct stat *)&buf64))) return rc;
   return XrdPosix_CopyStat(buf, buf64);
#endif
}
}
#endif

/******************************************************************************/
/*                                s t a t f s                                 */
/******************************************************************************/

#if !defined(__solaris__) && !defined(__macos__) && !defined(__FreeBSD__)
extern "C"
{
int        statfs(         const char *path, struct statfs *buf)
{
   static int Init = Xunix.Init(&Init);
   struct statfs64 buf64;
   int rc;

   if ((rc = XrdPosix_Statfs(path, (struct statfs *)&buf64))) return rc;
   memset(buf, 0, sizeof(buf));
   buf->f_type    = buf64.f_type;
   buf->f_bsize   = buf64.f_bsize;
   buf->f_blocks  = buf64.f_blocks;
   buf->f_bfree   = buf64.f_bfree;
   buf->f_files   = buf64.f_files;
   buf->f_ffree   = buf64.f_ffree;
   buf->f_fsid    = buf64.f_fsid;
   buf->f_namelen = buf64.f_namelen;
   return 0;
}
}
#endif

/******************************************************************************/
/*                               s t a t v f s                                */
/******************************************************************************/

#if !defined(__macos__) && !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
int        statvfs(         const char *path, struct statvfs *buf)
{
   static int Init = Xunix.Init(&Init);
   struct statvfs64 buf64;
   int rc;
   if ((rc = XrdPosix_Statvfs(path, (struct statvfs *)&buf64))) return rc;
   memset(buf, 0, sizeof(buf));
   buf->f_flag    = buf64.f_flag;
   buf->f_bsize   = buf64.f_bsize;
   buf->f_blocks  = buf64.f_blocks;
   buf->f_bfree   = buf64.f_bfree;
   buf->f_files   = buf64.f_files;
   buf->f_ffree   = buf64.f_ffree;
   buf->f_fsid    = buf64.f_fsid;
   buf->f_namemax = buf64.f_namemax;
   return 0;
}
}
#endif

/******************************************************************************/
/*                              t r u n c a t e                               */
/******************************************************************************/
  
#if !defined(SUNX86) && !defined(__FreeBSD__)
extern "C"
{
int truncate(const char *path, off_t offset)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Truncate(path, offset);
}
}
#endif
