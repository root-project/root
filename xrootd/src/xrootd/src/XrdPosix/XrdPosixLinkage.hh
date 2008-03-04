#ifndef _XRDPOSIXLINKAGE_H_
#define _XRDPOSIXLINKAGE_H_
/******************************************************************************/
/*                                                                            */
/*                    X r d P o s i x L i n k a g e . h h                     */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

/******************************************************************************/
/*                Posix Symbols vs Return Valus and Arguments                 */
/******************************************************************************/

//#ifdef __linux__
//#define UNIX_PFX "__"
//#else
#define UNIX_PFX
//#endif

#ifdef __macos__
#define off64_t int64_t
#define ELIBACC ESHLIBVERS
#endif
  
#define Symb_Access UNIX_PFX "access"
#define Retv_Access int
#define Args_Access const char *path, int amode
  
#define Symb_Chdir UNIX_PFX "chdir"
#define Retv_Chdir int
#define Args_Chdir const char *path
  
#define Symb_Close UNIX_PFX "close"
#define Retv_Close int
#define Args_Close int

#define Symb_Closedir UNIX_PFX "closedir"
#define Retv_Closedir int
#define Args_Closedir DIR *

#define Symb_Fclose UNIX_PFX "fclose"
#define Retv_Fclose int
#define Args_Fclose FILE *

#define Symb_Fcntl   UNIX_PFX "fcntl"
#define Retv_Fcntl   int
#define Args_Fcntl   int, int, ...

#define Symb_Fcntl64 UNIX_PFX "fcntl64"
#define Retv_Fcntl64 int
#define Args_Fcntl64 int, int, ...

#define Symb_Fdatasync UNIX_PFX "fdatasync"
#define Retv_Fdatasync int
#define Args_Fdatasync int

#define Symb_Fopen UNIX_PFX "fopen"
#define Retv_Fopen FILE *
#define Args_Fopen const char *, const char *

#define Symb_Fopen64 UNIX_PFX "fopen64"
#define Retv_Fopen64 FILE *
#define Args_Fopen64 const char *, const char *

#ifdef __linux__
#define Symb_Fstat UNIX_PFX "__fxstat"
#define Retv_Fstat int
#define Args_Fstat int, int, struct stat *
#else
#define Symb_Fstat UNIX_PFX "fstat"
#define Retv_Fstat int
#define Args_Fstat int, struct stat *
#endif

#ifdef __linux__
#define Symb_Fstat64 UNIX_PFX "__fxstat64"
#define Retv_Fstat64 int
#define Args_Fstat64 int, int, struct stat64 *
#else
#define Symb_Fstat64 UNIX_PFX "fstat64"
#define Retv_Fstat64 int
#define Args_Fstat64 int, struct stat64 *
#endif

#define Symb_Fsync UNIX_PFX "fsync"
#define Retv_Fsync int
#define Args_Fsync int

#define Symb_Fgetxattr UNIX_PFX "fgetxattr"
#define Retv_Fgetxattr ssize_t
#define Args_Fgetxattr int, const char *, const void *, size_t

#define Symb_Getxattr UNIX_PFX "getxattr"
#define Retv_Getxattr ssize_t
#define Args_Getxattr const char *, const char *, const void *, size_t

#define Symb_Lgetxattr UNIX_PFX "lgetxattr"
#define Retv_Lgetxattr ssize_t
#define Args_Lgetxattr const char *, const char *, const void *, size_t

#define Symb_Lseek UNIX_PFX "lseek"
#define Retv_Lseek off_t
#define Args_Lseek int, off_t, int

#define Symb_Lseek64 UNIX_PFX "lseek64"
#define Retv_Lseek64 off64_t
#define Args_Lseek64 int, off64_t, int

#ifdef __linux__
#define Symb_Lstat UNIX_PFX "__lxstat"
#define Retv_Lstat int
#define Args_Lstat int, const char *, struct stat *
#else
#define Symb_Lstat UNIX_PFX "lstat"
#define Retv_Lstat int
#define Args_Lstat const char *, struct stat *
#endif

#ifdef __linux__
#define Symb_Lstat64 UNIX_PFX "__lxstat64"
#define Retv_Lstat64 int
#define Args_Lstat64 int, const char *, struct stat64 *
#else
#define Symb_Lstat64 UNIX_PFX "lstat64"
#define Retv_Lstat64 int
#define Args_Lstat64 const char *, struct stat64 *
#endif

#define Symb_Mkdir UNIX_PFX "mkdir"
#define Retv_Mkdir int
#define Args_Mkdir const char *, mode_t

#define Symb_Open UNIX_PFX "open"
#define Retv_Open int
#define Args_Open const char *, int, ...

#define Symb_Open64 UNIX_PFX "open64"
#define Retv_Open64 int
#define Args_Open64 const char *, int, ...

#define Symb_Opendir UNIX_PFX "opendir"
#define Retv_Opendir DIR *
#define Args_Opendir const char *
  
#define Symb_Pread UNIX_PFX "pread"
#define Retv_Pread ssize_t
#define Args_Pread int, void *, size_t, off_t
  
#define Symb_Pread64 UNIX_PFX "pread64"
#define Retv_Pread64 ssize_t
#define Args_Pread64 int, void *, size_t, off64_t

#define Symb_Read UNIX_PFX "read"
#define Retv_Read ssize_t
#define Args_Read int, void *, size_t
  
#define Symb_Readv UNIX_PFX "readv"
#define Retv_Readv ssize_t
#define Args_Readv int, const struct iovec *, int

#define Symb_Readdir UNIX_PFX "readdir"
#define Retv_Readdir struct dirent *
#define Args_Readdir DIR *

#define Symb_Readdir64 UNIX_PFX "readdir64"
#define Retv_Readdir64 struct dirent64 *
#define Args_Readdir64 DIR *

#define Symb_Readdir_r UNIX_PFX "readdir_r"
#define Retv_Readdir_r int
#define Args_Readdir_r DIR *, struct dirent *, struct dirent **

#define Symb_Readdir64_r UNIX_PFX "readdir64_r"
#define Retv_Readdir64_r int
#define Args_Readdir64_r DIR *, struct dirent64 *, struct dirent64 **

#define Symb_Rename UNIX_PFX "rename"
#define Retv_Rename int
#define Args_Rename const char *, const char *

#define Symb_Rewinddir UNIX_PFX "rewinddir"
#define Retv_Rewinddir void
#define Args_Rewinddir DIR *

#define Symb_Rmdir UNIX_PFX "rmdir"
#define Retv_Rmdir int
#define Args_Rmdir const char *

#define Symb_Seekdir UNIX_PFX "seekdir"
#define Retv_Seekdir void
#define Args_Seekdir DIR *, long

#ifdef __linux__
#define Symb_Stat UNIX_PFX "__xstat"
#define Retv_Stat int
#define Args_Stat int, const char *, struct stat *
#else
#define Symb_Stat UNIX_PFX "stat"
#define Retv_Stat int
#define Args_Stat const char *, struct stat *
#endif

#ifdef __linux__
#define Symb_Stat64 UNIX_PFX "__xstat64"
#define Retv_Stat64 int
#define Args_Stat64 int, const char *, struct stat64 *
#else
#define Symb_Stat64 UNIX_PFX "stat64"
#define Retv_Stat64 int
#define Args_Stat64 const char *, struct stat64 *
#endif

#define Symb_Pwrite UNIX_PFX "pwrite"
#define Retv_Pwrite ssize_t
#define Args_Pwrite int, const void *, size_t, off_t

#define Symb_Pwrite64 UNIX_PFX "pwrite64"
#define Retv_Pwrite64 ssize_t
#define Args_Pwrite64 int, const void *, size_t, off64_t

#define Symb_Telldir UNIX_PFX "telldir"
#define Retv_Telldir long
#define Args_Telldir DIR *

#define Symb_Unlink UNIX_PFX "unlink"
#define Retv_Unlink int
#define Args_Unlink const char *

#define Symb_Write UNIX_PFX "write"
#define Retv_Write ssize_t
#define Args_Write int, const void *, size_t

#define Symb_Writev UNIX_PFX "writev"
#define Retv_Writev ssize_t
#define Args_Writev int, const struct iovec *, int

/******************************************************************************/
/*            C a l l   O u t   V e c t o r   D e f i n i t i o n             */
/******************************************************************************/
  
class XrdPosixLinkage
{public:
      int              Init(int *x) {return (Done ? 0 : Resolve());}

      Retv_Access      (*Access)(Args_Access);
      Retv_Chdir       (*Chdir)(Args_Chdir);
      Retv_Close       (*Close)(Args_Close);
      Retv_Closedir    (*Closedir)(Args_Closedir);
      Retv_Fclose      (*Fclose)(Args_Fclose);
      Retv_Fcntl       (*Fcntl)(Args_Fcntl);
      Retv_Fcntl64     (*Fcntl64)(Args_Fcntl64);
      Retv_Fdatasync   (*Fdatasync)(Args_Fdatasync);
      Retv_Fopen       (*Fopen)(Args_Fopen);
      Retv_Fopen64     (*Fopen64)(Args_Fopen64);
      Retv_Fstat       (*Fstat)(Args_Fstat);
      Retv_Fstat64     (*Fstat64)(Args_Fstat64);
      Retv_Fsync       (*Fsync)(Args_Fsync);
      Retv_Fgetxattr   (*Fgetxattr)(Args_Fgetxattr);
      Retv_Lgetxattr   (*Lgetxattr)(Args_Lgetxattr);
      Retv_Getxattr    (*Getxattr)(Args_Getxattr);
      Retv_Lseek       (*Lseek)(Args_Lseek);
      Retv_Lseek64     (*Lseek64)(Args_Lseek64);
      Retv_Lstat       (*Lstat)(Args_Lstat);
      Retv_Lstat64     (*Lstat64)(Args_Lstat64);
      Retv_Mkdir       (*Mkdir)(Args_Mkdir);
      Retv_Open        (*Open)(Args_Open);
      Retv_Open64      (*Open64)(Args_Open64);
      Retv_Opendir     (*Opendir)(Args_Opendir);
      Retv_Pread       (*Pread)(Args_Pread);
      Retv_Pread64     (*Pread64)(Args_Pread64);
      Retv_Read        (*Read)(Args_Read);
      Retv_Readv       (*Readv)(Args_Readv);
      Retv_Readdir     (*Readdir)(Args_Readdir);
      Retv_Readdir64   (*Readdir64)(Args_Readdir64);
      Retv_Readdir_r   (*Readdir_r)(Args_Readdir_r);
      Retv_Readdir64_r (*Readdir64_r)(Args_Readdir64_r);
      Retv_Rename      (*Rename)(Args_Rename);
      Retv_Rewinddir   (*Rewinddir)(Args_Rewinddir);
      Retv_Rmdir       (*Rmdir)(Args_Rmdir);
      Retv_Seekdir     (*Seekdir)(Args_Seekdir);
      Retv_Stat        (*Stat)(Args_Stat);
      Retv_Stat64      (*Stat64)(Args_Stat64);
      Retv_Pwrite      (*Pwrite)(Args_Pwrite);
      Retv_Pwrite64    (*Pwrite64)(Args_Pwrite64);
      Retv_Telldir     (*Telldir)(Args_Telldir);
      Retv_Unlink      (*Unlink)(Args_Unlink);
      Retv_Write       (*Write)(Args_Write);
      Retv_Writev      (*Writev)(Args_Writev);

      int              Load_Error(const char *epname, int retv=-1);

      XrdPosixLinkage() {Init(0);}
     ~XrdPosixLinkage() {}

private:
int Done;
int Resolve();
};

/******************************************************************************/
/*          X r d P o s i x   L i n k a g e   D e f i n i t i o n s           */
/******************************************************************************/
  
#define XRD_Retv_Access int
#define XRD_Args_Access const char *path, int amode
  
#define XRD_Retv_Chdir int
#define XRD_Args_Chdir const char *path
  
#define XRD_Retv_Close int
#define XRD_Args_Close int

#define XRD_Retv_Closedir int
#define XRD_Args_Closedir DIR *

#define XRD_Retv_Lseek off64_t
#define XRD_Args_Lseek int, off64_t, int

#define XRD_Retv_Fcntl int
#define XRD_Args_Fcntl int, int, ...

#define XRD_Retv_Fstat int
#define XRD_Args_Fstat int, struct stat *

#define XRD_Retv_Fsync int
#define XRD_Args_Fsync int

#define XRD_Retv_Lstat int
#define XRD_Args_Lstat const char *, struct stat *

#define XRD_Retv_Mkdir int
#define XRD_Args_Mkdir const char *, mode_t

#define XRD_Retv_Open int
#define XRD_Args_Open const char *, int, ...

#define XRD_Retv_Opendir DIR *
#define XRD_Args_Opendir const char *
  
#define XRD_Retv_Pread ssize_t
#define XRD_Args_Pread int, void *, size_t, off64_t

#define XRD_Retv_Read ssize_t
#define XRD_Args_Read int, void *, size_t
  
#define XRD_Retv_Readv ssize_t
#define XRD_Args_Readv int, const struct iovec *, int

#define XRD_Retv_Readdir     struct dirent *
#define XRD_Args_Readdir     DIR *

#define XRD_Retv_Readdir64   struct dirent64 *
#define XRD_Args_Readdir64   DIR *

#define XRD_Retv_Readdir_r   int
#define XRD_Args_Readdir_r   DIR *, struct dirent   *, struct dirent   **

#define XRD_Retv_Readdir64_r int
#define XRD_Args_Readdir64_r DIR *, struct dirent64 *, struct dirent64 **

#define XRD_Retv_Rename int
#define XRD_Args_Rename const char *, const char *

#define XRD_Retv_Rewinddir void
#define XRD_Args_Rewinddir DIR *

#define XRD_Retv_Rmdir int
#define XRD_Args_Rmdir const char *

#define XRD_Retv_Seekdir void
#define XRD_Args_Seekdir DIR *, long

#define XRD_Retv_Stat int
#define XRD_Args_Stat const char *, struct stat *

#define XRD_Retv_Pwrite ssize_t
#define XRD_Args_Pwrite int, const void *, size_t, off64_t

#define XRD_Retv_Telldir long
#define XRD_Args_Telldir DIR *

#define XRD_Retv_Unlink int
#define XRD_Args_Unlink const char *

#define XRD_Retv_Write ssize_t
#define XRD_Args_Write int, const void *, size_t

#define XRD_Retv_Writev ssize_t
#define XRD_Args_Writev int, const struct iovec *, int

#define XRD_Retv_isMyPath int
#define XRD_Args_isMyPath const char *

/******************************************************************************/
/*                       X r d P o s i x R o o t V e c                        */
/******************************************************************************/
  
class XrdPosixRootVec
{public:
      int                  Init(int *x) {return (Done ? 0 : Resolve());}

      XRD_Retv_Access      (*Access)(XRD_Args_Access);
      XRD_Retv_Chdir       (*Chdir)(XRD_Args_Chdir);
      XRD_Retv_Close       (*Close)(XRD_Args_Close);
      XRD_Retv_Closedir    (*Closedir)(XRD_Args_Closedir);
      XRD_Retv_Lseek       (*Lseek)(XRD_Args_Lseek);
      XRD_Retv_Fcntl       (*Fcntl)(XRD_Args_Fcntl);
      XRD_Retv_Fstat       (*Fstat)(XRD_Args_Fstat);
      XRD_Retv_Fsync       (*Fsync)(XRD_Args_Fsync);
      XRD_Retv_Lstat       (*Lstat)(XRD_Args_Lstat);
      XRD_Retv_Mkdir       (*Mkdir)(XRD_Args_Mkdir);
      XRD_Retv_Open        (*Open)(XRD_Args_Open);
      XRD_Retv_Opendir     (*Opendir)(XRD_Args_Opendir);
      XRD_Retv_Pread       (*Pread)(XRD_Args_Pread);
      XRD_Retv_Read        (*Read)(XRD_Args_Read);
      XRD_Retv_Readv       (*Readv)(XRD_Args_Readv);
      XRD_Retv_Readdir     (*Readdir)(XRD_Args_Readdir);
      XRD_Retv_Readdir64   (*Readdir64)(XRD_Args_Readdir64);
      XRD_Retv_Readdir_r   (*Readdir_r)(XRD_Args_Readdir_r);
      XRD_Retv_Readdir64_r (*Readdir64_r)(XRD_Args_Readdir64_r);
      XRD_Retv_Rename      (*Rename)(XRD_Args_Rename);
      XRD_Retv_Rewinddir   (*Rewinddir)(XRD_Args_Rewinddir);
      XRD_Retv_Rmdir       (*Rmdir)(XRD_Args_Rmdir);
      XRD_Retv_Seekdir     (*Seekdir)(XRD_Args_Seekdir);
      XRD_Retv_Stat        (*Stat)(XRD_Args_Stat);
      XRD_Retv_Pwrite      (*Pwrite)(XRD_Args_Pwrite);
      XRD_Retv_Telldir     (*Telldir)(XRD_Args_Telldir);
      XRD_Retv_Unlink      (*Unlink)(XRD_Args_Unlink);
      XRD_Retv_Write       (*Write)(XRD_Args_Write);
      XRD_Retv_Writev      (*Writev)(XRD_Args_Writev);
      XRD_Retv_isMyPath    (*isMyPath)(XRD_Args_isMyPath);

      XrdPosixRootVec() {Init(0);}
     ~XrdPosixRootVec() {}

private:
int Done;
int Resolve();
};
// Warning! This class is meant to be defined as a static object.
#endif
