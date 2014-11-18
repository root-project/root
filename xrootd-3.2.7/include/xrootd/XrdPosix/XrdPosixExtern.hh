#ifndef __XRDPOSIXEXTERN_H__
#define __XRDPOSIXEXTERN_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d P o s i x E x t e r n . h h                      */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/* Modified by Frank Winklmeier to add the full Posix file system definition. */
/******************************************************************************/
  
//           $Id$

// These OS-Compatible (not C++) externs are included by XrdPosix.hh to
// complete the macro definitions contained therein.

// Use this file directly to define your own macros or interfaces. Note that
// native types are used to avoid 32/64 bit parameter/return value ambiguities
// and to enforce shared library compatability (needed by the preload32 code).

// Only 64-bit interfaces are directly supported. However, the preload library
// supports the old 32-bit interfaces. To use this include you must specify

// -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64

// compilation options. This ensures LP64 compatability which defines:
//
// ssize_t ->          long long
//  size_t -> unsigned long long
//   off_t ->          long long

#if (!defined(_LARGEFILE_SOURCE) || !defined(_LARGEFILE64_SOURCE) || \
    _FILE_OFFSET_BITS!=64) && !defined(XRDPOSIXPRELOAD32)
#error Compilation options are incompatible with XrdPosixExtern.hh; \
       Specify -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64
#endif

// We pre-declare various structure t avoid compilation complaints. We cannot
// include the necessary ".h" files as these would also try to define entry
// points which may conflict with our definitions due to renaming pragmas and
// simple defines. All we want is to make sure we have the right name in the
// loader's symbol table so that the preload library can intercept the call.
// We need these definitions here because the includer may not have included
// all of the includes necessary to support all of the API's.
//
struct iovec;
struct stat;
struct statfs;
struct statvfs;

#include <dirent.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

#include "XrdPosix/XrdPosixOsDep.hh"

#ifdef __cplusplus
extern "C"
{
#endif
extern int        XrdPosix_Access(const char *path, int amode);

extern int        XrdPosix_Acl(const char *path, int cmd, int nentries,
                               void *aclbufp);

extern int        XrdPosix_Chdir(const char *path);

extern int        XrdPosix_Close(int fildes);

extern int        XrdPosix_Closedir(DIR *dirp);

extern int        XrdPosix_Creat(const char *path, mode_t mode);

extern int        XrdPosix_Fclose(FILE *stream);

extern int        XrdPosix_Fcntl(int fd, int cmd, ...);

extern int        XrdPosix_Fdatasync(int fildes);

extern int        XrdPosix_Fflush(FILE *stream);

#ifdef __linux__
extern long long  XrdPosix_Fgetxattr (int fd, const char *name,
                                      void *value, unsigned long long size);
#endif

extern FILE      *XrdPosix_Fopen(const char *path, const char *mode);

extern size_t     XrdPosix_Fread(void *ptr, size_t size, size_t nitems, FILE *stream);

extern int        XrdPosix_Fseek(FILE *stream, long offset, int whence);

extern int        XrdPosix_Fseeko(FILE *stream, long long offset, int whence);

extern int        XrdPosix_Fstat(int fildes, struct stat *buf);

#ifdef __linux__
extern int        XrdPosix_FstatV(int ver, int fildes, struct stat *buf);
#endif

extern int        XrdPosix_Fsync(int fildes);

extern long       XrdPosix_Ftell(FILE *stream);

extern long long  XrdPosix_Ftello(FILE *stream);

extern int        XrdPosix_Ftruncate(int fildes, long long offset);

extern size_t     XrdPosix_Fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream);

#ifdef __linux__
extern long long  XrdPosix_Getxattr (const char *path, const char *name, 
                                     void *value, unsigned long long size);

extern long long  XrdPosix_Lgetxattr(const char *path, const char *name, 
                                     void *value, unsigned long long size);
#endif

extern long long  XrdPosix_Lseek(int fildes, long long offset, int whence);

extern int        XrdPosix_Lstat(const char *path, struct stat *buf);

extern int        XrdPosix_Mkdir(const char *path, mode_t mode);

extern int        XrdPosix_Open(const char *path, int oflag, ...);

extern DIR*       XrdPosix_Opendir(const char *path);
  
extern long       XrdPosix_Pathconf(const char *path, int name);

extern long long  XrdPosix_Pread(int fildes, void *buf, unsigned long long nbyte,
                                 long long offset);

extern long long  XrdPosix_Read(int fildes, void *buf, unsigned long long nbyte);
  
extern long long  XrdPosix_Readv(int fildes, const struct iovec *iov, int iovcnt);

extern struct dirent*   XrdPosix_Readdir  (DIR *dirp);
extern struct dirent64* XrdPosix_Readdir64(DIR *dirp);

extern int        XrdPosix_Readdir_r  (DIR *dirp, struct dirent   *entry, struct dirent   **result);
extern int        XrdPosix_Readdir64_r(DIR *dirp, struct dirent64 *entry, struct dirent64 **result);

extern int        XrdPosix_Rename(const char *oname, const char *nname);

extern void       XrdPosix_Rewinddir(DIR *dirp);

extern int        XrdPosix_Rmdir(const char *path);

extern void       XrdPosix_Seekdir(DIR *dirp, long loc);

extern int        XrdPosix_Stat(const char *path, struct stat *buf);

#if !defined(__solaris__)
extern int        XrdPosix_Statfs(const char *path, struct statfs *buf);
#endif

extern int        XrdPosix_Statvfs(const char *path, struct statvfs *buf);

extern long long  XrdPosix_Pwrite(int fildes, const void *buf, 
                                  unsigned long long nbyte, long long offset);

extern long       XrdPosix_Telldir(DIR *dirp);

extern int        XrdPosix_Truncate(const char *path, long long offset);

extern int        XrdPosix_Unlink(const char *path);

extern long long  XrdPosix_Write(int fildes, const void *buf,
                                 unsigned long long nbyte);

extern long long  XrdPosix_Writev(int fildes, const struct iovec *iov, int iovcnt);

#ifdef __cplusplus
};
#endif

// The following is for use for wrapper classeses
//
extern int        XrdPosix_isMyPath(const char *path);

extern char      *XrdPosix_URL(const char *path, char *buff, int blen);

#endif
