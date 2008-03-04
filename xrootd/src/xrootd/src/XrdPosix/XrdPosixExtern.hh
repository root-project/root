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

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

// We need to avoid using dirent64 for MacOS platforms. We would normally
// include XrdSysPlatform.hh for this but this include file needs to be
// standalone. So, we replicate the dirent64 redefinition here,
//
#if defined(__macos__) && !defined(dirent64)
#define dirent64 dirent
#endif

// Define the external interfaces (not C++ but OS compatabile). These
// externs are included by XrdPosix.hh to complete the macro definitions.
// Use this file directly to define your own macros or interfaces.
//
extern int     XrdPosix_Chdir(const char *path);

extern int     XrdPosix_Close(int fildes);

extern int     XrdPosix_Closedir(DIR *dirp);

extern off_t   XrdPosix_Lseek(int fildes, off_t offset, int whence);

extern int     XrdPosix_Fstat(int fildes, struct stat *buf);

extern int     XrdPosix_Fsync(int fildes);

extern int     XrdPosix_Mkdir(const char *path, mode_t mode);

extern int     XrdPosix_Open(const char *path, int oflag, ...);

extern DIR*    XrdPosix_Opendir(const char *path);
  
extern ssize_t XrdPosix_Pread(int fildes, void *buf, size_t nbyte, off_t offset);

extern ssize_t XrdPosix_Read(int fildes, void *buf, size_t nbyte);
  
extern ssize_t XrdPosix_Readv(int fildes, const struct iovec *iov, int iovcnt);

extern struct  dirent*   XrdPosix_Readdir  (DIR *dirp);
extern struct  dirent64* XrdPosix_Readdir64(DIR *dirp);

extern int     XrdPosix_Readdir_r  (DIR *dirp, struct dirent   *entry, struct dirent   **result);
extern int     XrdPosix_Readdir64_r(DIR *dirp, struct dirent64 *entry, struct dirent64 **result);

extern int     XrdPosix_Rename(const char *oname, const char *nname);

extern void    XrdPosix_Rewinddir(DIR *dirp);

extern int     XrdPosix_Rmdir(const char *path);

extern void    XrdPosix_Seekdir(DIR *dirp, long loc);

extern int     XrdPosix_Stat(const char *path, struct stat *buf);

extern ssize_t XrdPosix_Pwrite(int fildes, const void *buf, size_t nbyte, off_t offset);

extern long    XrdPosix_Telldir(DIR *dirp);

extern int     XrdPosix_Unlink(const char *path);

extern ssize_t XrdPosix_Write(int fildes, const void *buf, size_t nbyte);

extern ssize_t XrdPosix_Writev(int fildes, const struct iovec *iov, int iovcnt);

// The following is for use for wrapper classeses
//
extern int     XrdPosix_isMyPath(const char *path);

#endif
