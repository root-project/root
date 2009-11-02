/******************************************************************************/
/*                                                                            */
/*                    X r d P o s i x P r e l o a d . c c                     */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdPosixPreloadCVSID = "$Id$";

#include <sys/types.h>
#include <stdarg.h>
#include <unistd.h>

#include "XrdPosix/XrdPosixLinkage.hh"
#include "XrdPosix/XrdPosixOsDep.hh"

/******************************************************************************/
/*                      P r e - D e c l a r a t i o n s                       */
/******************************************************************************/

#include "XrdPosix/XrdPosixExtern.hh"
 
/******************************************************************************/
/*                   G l o b a l   D e c l a r a t i o n s                    */
/******************************************************************************/
  
extern XrdPosixLinkage Xunix;
  
/******************************************************************************/
/*                                a c c e s s                                 */
/******************************************************************************/
  
extern "C"
{
int access(const char *path, int amode)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Access(path, amode);
}
}

/******************************************************************************/
/*                                   a c l                                    */
/******************************************************************************/

// This is a required addition for Solaris 10+ systems

extern "C"
{
int acl(const char *path, int cmd, int nentries, void *aclbufp)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Acl(path, cmd, nentries, aclbufp);
}
}
  
/******************************************************************************/
/*                                 c h d i r                                  */
/******************************************************************************/

extern "C"
{
int     chdir(const char *path)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Chdir(path);
}
}

/******************************************************************************/
/*                                 c l o s e                                  */
/******************************************************************************/

extern "C"
{
int     close(int fildes)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Close(fildes);
}
}

/******************************************************************************/
/*                              c l o s e d i r                               */
/******************************************************************************/
  
extern "C"
{
int     closedir(DIR *dirp)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Closedir(dirp);
}
}

/******************************************************************************/
/*                                 c r e a t                                  */
/******************************************************************************/
  
extern "C"
{
int     creat64(const char *path, mode_t mode)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Creat(path, mode);
}
}
  
/******************************************************************************/
/*                                f c l o s e                                 */
/******************************************************************************/

extern "C"
{
int fclose(FILE *stream)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fclose(stream);
}
}

/******************************************************************************/
/*                               f c n t l 6 4                                */
/******************************************************************************/
  
extern "C"
{
int     fcntl64(int fd, int cmd, ...)
{
   static int Init = Xunix.Init(&Init);
   va_list ap;
   void *theArg;

   va_start(ap, cmd);
   theArg = va_arg(ap, void *);
   va_end(ap);
   return XrdPosix_Fcntl(fd, cmd, theArg);
}
}

/******************************************************************************/
/*                             f d a t a s y n c                              */
/******************************************************************************/
  
extern "C"
{
int     fdatasync(int fildes)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fdatasync(fildes);
}
}

/******************************************************************************/
/*                                f f l u s h                                 */
/******************************************************************************/
  
extern "C"
{
int    fflush(FILE *stream)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fflush(stream);
}
}
  
/******************************************************************************/
/*                                 f o p e n                                  */
/******************************************************************************/
  
extern "C"
{
FILE  *fopen64(const char *path, const char *mode)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fopen(path, mode);
}
}

/******************************************************************************/
/*                                 f r e a d                                  */
/******************************************************************************/
  
extern "C"
{
size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fread(ptr, size, nitems, stream);
}
}
  
/******************************************************************************/
/*                                 f s e e k                                  */
/******************************************************************************/

extern "C"
{
int fseek(FILE *stream, long offset, int whence)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fseek(stream, offset, whence);
}
}
  
/******************************************************************************/
/*                                f s e e k o                                 */
/******************************************************************************/

extern "C"
{
int fseeko64(FILE *stream, off64_t offset, int whence)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fseeko(stream, offset, whence);
}
}
  
/******************************************************************************/
/*                                 f s t a t                                  */
/******************************************************************************/

extern "C"
{
#if defined __linux__ && __GNUC__ && __GNUC__ >= 2
int  __fxstat64(int ver, int fildes, struct stat64 *buf)
#else
int     fstat64(         int fildes, struct stat64 *buf)
#endif
{
   static int Init = Xunix.Init(&Init);

#ifdef __linux__
   return XrdPosix_FstatV(ver, fildes, (struct stat *)buf);
#else
   return XrdPosix_Fstat (     fildes, (struct stat *)buf);
#endif
}
}

/******************************************************************************/
/*                                 f s y n c                                  */
/******************************************************************************/
  
extern "C"
{
int     fsync(int fildes)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fsync(fildes);
}
}
  
/******************************************************************************/
/*                                 f t e l l                                  */
/******************************************************************************/

extern "C"
{
long    ftell(FILE *stream)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Ftell(stream);
}
}
  
/******************************************************************************/
/*                                f t e l l o                                 */
/******************************************************************************/

extern "C"
{
off64_t ftello64(FILE *stream)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Ftello(stream);
}
}
  
/******************************************************************************/
/*                             f t r u n c a t e                              */
/******************************************************************************/
  
extern "C"
{
int ftruncate64(int fildes, off_t offset)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Ftruncate(fildes, offset);
}
}
  
/******************************************************************************/
/*                                f w r i t e                                 */
/******************************************************************************/
  
extern "C"
{
size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fwrite(ptr, size, nitems, stream);
}
}
  
/******************************************************************************/
/*                             f g e t x a t t r                              */
/******************************************************************************/
  
#ifdef __linux__
extern "C"
{
ssize_t fgetxattr (int fd, const char *name, void *value, size_t size)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Fgetxattr(fd, name, value, size);
}
}
#endif

/******************************************************************************/
/*                              g e t x a t t r                               */
/******************************************************************************/
  
#ifdef __linux__
extern "C"
{
ssize_t getxattr (const char *path, const char *name, void *value, size_t size)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Getxattr(path, name, value, size);
}
}
#endif
  
/******************************************************************************/
/*                             l g e t x a t t r                              */
/******************************************************************************/
  
#ifdef __linux__
extern "C"
{
ssize_t lgetxattr (const char *path, const char *name, void *value, size_t size)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Lgetxattr(path, name, value, size);
}
}
#endif

/******************************************************************************/
/*                                 l s e e k                                  */
/******************************************************************************/
  
extern "C"
{
off64_t lseek64(int fildes, off64_t offset, int whence)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Lseek(fildes, offset, whence);
}
}

/******************************************************************************/
/*                                l l s e e k                                 */
/******************************************************************************/
  
extern "C"
{
#if defined(__linux__) || defined(__macos__)
off_t      llseek(int fildes, off_t    offset, int whence)
#else
offset_t   llseek(int fildes, offset_t offset, int whence)
#endif
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Lseek(fildes, offset, whence);
}
}

/******************************************************************************/
/*                                 l s t a t                                  */
/******************************************************************************/

extern "C"
{
#if defined __linux__ && __GNUC__ && __GNUC__ >= 2
int     __lxstat64(int ver, const char *path, struct stat64 *buf)
#else
int        lstat64(         const char *path, struct stat64 *buf)
#endif
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Lstat(path, (struct stat *)buf);
}
}

/******************************************************************************/
/*                                 m k d i r                                  */
/******************************************************************************/
  
extern "C"
{
int     mkdir(const char *path, mode_t mode)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Mkdir(path, mode);
}
}

/******************************************************************************/
/*                                  o p e n                                   */
/******************************************************************************/

extern "C"
{
int     open64(const char *path, int oflag, ...)
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

/******************************************************************************/
/*                               o p e n d i r                                */
/******************************************************************************/
  
extern "C"
{
DIR*    opendir(const char *path)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Opendir(path);
}
}
  
/******************************************************************************/
/*                              p a t h c o n f                               */
/******************************************************************************/

// This is a required addition for Solaris 10+ systems

extern "C"
{
long pathconf(const char *path, int name)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Pathconf(path, name);
}
}

/******************************************************************************/
/*                                 p r e a d                                  */
/******************************************************************************/
  
extern "C"
{
ssize_t pread64(int fildes, void *buf, size_t nbyte, off_t offset)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Pread  (fildes, buf, nbyte, offset);
}
}

/******************************************************************************/
/*                                p w r i t e                                 */
/******************************************************************************/
  
extern "C"
{
ssize_t pwrite64(int fildes, const void *buf, size_t nbyte, off_t offset)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Pwrite(fildes, buf, nbyte, offset);
}
}

/******************************************************************************/
/*                                  r e a d                                   */
/******************************************************************************/
  
extern "C"
{
ssize_t read(int fildes, void *buf, size_t nbyte)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Read(fildes, buf, nbyte);
}
}
  
/******************************************************************************/
/*                                 r e a d v                                  */
/******************************************************************************/
  
extern "C"
{
ssize_t readv(int fildes, const struct iovec *iov, int iovcnt)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Readv(fildes, iov, iovcnt);
}
}

/******************************************************************************/
/*                               r e a d d i r                                */
/******************************************************************************/

extern "C"
{
struct dirent64* readdir64(DIR *dirp)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Readdir64(dirp);
}
}

/******************************************************************************/
/*                             r e a d d i r _ r                              */
/******************************************************************************/
  
extern "C"
{
int     readdir64_r(DIR *dirp, struct dirent64 *entry, struct dirent64 **result)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Readdir64_r(dirp, entry, result);
}
}

/******************************************************************************/
/*                                r e n a m e                                 */
/******************************************************************************/
  
extern "C"
{
int     rename(const char *oldpath, const char *newpath)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Rename(oldpath, newpath);
}
}

/******************************************************************************/
/*                             r e w i n d d i r                              */
/******************************************************************************/

#ifndef rewinddir
extern "C"
{
void    rewinddir(DIR *dirp)
{
   static int Init = Xunix.Init(&Init);

   XrdPosix_Rewinddir(dirp);
}
}
#endif

/******************************************************************************/
/*                                 r m d i r                                  */
/******************************************************************************/
  
extern "C"
{
int     rmdir(const char *path)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Rmdir(path);
}
}

/******************************************************************************/
/*                               s e e k d i r                                */
/******************************************************************************/
  
extern "C"
{
void    seekdir(DIR *dirp, long loc)
{
   static int Init = Xunix.Init(&Init);

   XrdPosix_Seekdir(dirp, loc);
}
}

/******************************************************************************/
/*                                  s t a t                                   */
/******************************************************************************/

extern "C"
{
#if defined __linux__ && __GNUC__ && __GNUC__ >= 2
int     __xstat64(int ver, const char *path, struct stat64 *buf)
#else
int        stat64(         const char *path, struct stat64 *buf)
#endif
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Stat(path, (struct stat *)buf);
}
}

/******************************************************************************/
/*                                s t a t f s                                 */
/******************************************************************************/

#if !defined(__solaris__)
extern "C"
{
int        statfs64(       const char *path, struct statfs64 *buf)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Statfs(path, (struct statfs *)buf);
}
}
#endif

/******************************************************************************/
/*                               s t a t v f s                                */
/******************************************************************************/

extern "C"
{
int        statvfs64(         const char *path, struct statvfs64 *buf)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Statvfs(path, (struct statvfs *)buf);
}
}

/******************************************************************************/
/*                               t e l l d i r                                */
/******************************************************************************/
  
extern "C"
{
long    telldir(DIR *dirp)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Telldir(dirp);
}
}
  
/******************************************************************************/
/*                              t r u n c a t e                               */
/******************************************************************************/
  
extern "C"
{
int truncate64(const char *path, off_t offset)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Truncate(path, offset);
}
}

/******************************************************************************/
/*                                u n l i n k                                 */
/******************************************************************************/
  
extern "C"
{
int     unlink(const char *path)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Unlink(path);
}
}

/******************************************************************************/
/*                                 w r i t e                                  */
/******************************************************************************/
  
extern "C"
{
ssize_t write(int fildes, const void *buf, size_t nbyte)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Write(fildes, buf, nbyte);
}
}

/******************************************************************************/
/*                                w r i t e v                                 */
/******************************************************************************/
  
extern "C"
{
ssize_t writev(int fildes, const struct iovec *iov, int iovcnt)
{
   static int Init = Xunix.Init(&Init);

   return XrdPosix_Writev(fildes, iov, iovcnt);
}
}
