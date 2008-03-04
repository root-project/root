/******************************************************************************/
/*                                                                            */
/*                           X r d P o s i x . c c                            */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

const char *XrdPosixCVSID = "$Id$";
  
#include <stdarg.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>
#include <iostream.h>

#include "XrdPosix/XrdPosixLinkage.hh"
#include "XrdPosix/XrdPosixXrootd.hh"
#include "XrdOuc/XrdOucTokenizer.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
class XrdPosixXrootPath
{
public:

void  CWD(const char *path);

char *URL(const char *path, char *buff, int blen);

      XrdPosixXrootPath();
     ~XrdPosixXrootPath();

private:

struct xpath 
       {struct xpath *next;
        const  char  *server;
               int    servln;
        const  char  *path;
               int    plen;
        const  char  *nath;
               int    nlen;

        xpath(struct xpath *cur,
              const   char *pServ,
              const   char *pPath,
              const   char *pNath) : next(cur),
                                     server(pServ),
                                     servln(strlen(pServ)),
                                     path(pPath),
                                     plen(strlen(pPath)),
                                     nath(pNath),
                                     nlen(pNath ? strlen(pNath) : 0) {}
       ~xpath() {}
       };

struct xpath *xplist;
char         *pBase;
char         *cwdPath;
int           cwdPlen;
};

/******************************************************************************/
/*         X r d P o s i x X r o o t P a t h   C o n s t r u c t o r          */
/******************************************************************************/

XrdPosixXrootPath::XrdPosixXrootPath()
    : xplist(0),
      pBase(0)
{
   XrdOucTokenizer thePaths(0);
   char *plist = 0, *colon = 0, *subs = 0, *lp = 0, *tp = 0;
   int aOK = 0;

   cwdPath = 0; cwdPlen = 0;

   if (!(plist = getenv("XROOTD_VMP")) || !*plist) return;
   pBase = strdup(plist);

   thePaths.Attach(pBase);

   if ((lp = thePaths.GetLine())) while((tp = thePaths.GetToken()))
      {aOK = 1;
       if ((colon = rindex(tp, (int)':')) && *(colon+1) == '/')
          {if (!(subs = index(colon, (int)'='))) subs = 0;
              else if (*(subs+1) == '/') {*subs = '\0'; subs++;}
                      else if (*(subs+1)) aOK = 0;
                              else {*subs = '\0'; subs = (char*)"";}
          } else aOK = 0;

       if (aOK)
          {*colon++ = '\0';
           while(*(colon+1) == '/') colon++;
           xplist = new xpath(xplist, tp, colon, subs);
          } else cerr <<"XrdPosix: Invalid XROOTD_VMP token '" <<tp <<'"' <<endl;
      }
}

/******************************************************************************/
/*          X r d P o s i x X r o o t P a t h   D e s t r u c t o r           */
/******************************************************************************/
  
XrdPosixXrootPath::~XrdPosixXrootPath()
{
   struct xpath *xpnow;

   while((xpnow = xplist))
        {xplist = xplist->next; delete xpnow;}
}
  
/******************************************************************************/
/*                     X r d P o s i x P a t h : : C W D                      */
/******************************************************************************/
  
void XrdPosixXrootPath::CWD(const char *path)
{
   if (cwdPath) free(cwdPath);
   cwdPlen = strlen(path);
   if (*(path+cwdPlen-1) == '/') cwdPath = strdup(path);
      else {char buff[2048];
            strcpy(buff, path); 
            *(buff+cwdPlen  ) = '/';
            *(buff+cwdPlen+1) = '\0';
            cwdPath = strdup(buff); cwdPlen++;
           }
}

/******************************************************************************/
/*                     X r d P o s i x P a t h : : U R L                      */
/******************************************************************************/
  
char *XrdPosixXrootPath::URL(const char *path, char *buff, int blen)
{
   const char   *rproto = "root://";
   const int     rprlen = strlen(rproto);
   const char   *xproto = "xroot://";
   const int     xprlen = strlen(xproto);
   struct xpath *xpnow = xplist;
   char tmpbuff[2048];
   int plen, pathlen = 0;

// If this starts with 'root", then this is our path
//
   if (!strncmp(rproto, path, rprlen)) return (char *)path;

// If it starts with xroot, then convert it to be root
//
   if (!strncmp(xproto, path, xprlen))
      {if (!buff) return (char *)1;
       if ((int(strlen(path))) > blen) return 0;
       strcpy(buff, path+1);
       return buff;
      }

// If a relative path was specified, convert it to an abso9lute path
//
   if (path[0] == '.' && path[1] == '/' && cwdPath)
      {pathlen = (strlen(path) + cwdPlen - 2);
       if (pathlen < (int)sizeof(tmpbuff))
          {strcpy(tmpbuff, cwdPath);
           strcpy(tmpbuff+cwdPlen, path+2);
           path = (const char *)tmpbuff;
          }  else return 0;
      }

// Check if this path starts with one or our known paths
//
   while(xpnow)
        if (!strncmp(path, xpnow->path, xpnow->plen)) break;
           else xpnow = xpnow->next;

// If we did not match a path, this is not our path.
//
   if (!xpnow) return 0;
   if (!buff) return (char *)1;

// Verify that we won't overflow the buffer
//
   if (!pathlen) pathlen = strlen(path);
   plen = xprlen + pathlen + xpnow->servln + 2;
   if (xpnow->nath) plen =  plen - xpnow->plen + xpnow->nlen;
   if (plen >= blen) return 0;

// Build the url
//
   strcpy(buff, rproto);
   strcat(buff, xpnow->server);
   strcat(buff, "/");
   if (xpnow->nath) {strcat(buff, xpnow->nath); path += xpnow->plen;}
   if (*path != '/') strcat(buff, "/");
   strcat(buff, path);
   return buff;
}

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/
  
static XrdPosixXrootd    Xroot;

static XrdPosixXrootPath XrootPath;
  
extern XrdPosixLinkage   Xunix;

/******************************************************************************/
/*                        X r d P o s i x _ C h d i r                         */
/******************************************************************************/

int XrdPosix_Chdir(const char *path)
{
   int rc;

// Set the working directory if the actual chdir succeeded
//
   if (!(rc = Xunix.Chdir(path))) XrootPath.CWD(path);
   return rc;
}
  
/******************************************************************************/
/*                        X r d P o s i x _ C l o s e                         */
/******************************************************************************/

int XrdPosix_Close(int fildes)
{

// Return result of the close
//
   return (fildes < XrdPosixFD ? Xunix.Close(fildes) : Xroot.Close(fildes));
}

/******************************************************************************/
/*                       X r d P o s i x _ A c c e s s                        */
/******************************************************************************/
  
int XrdPosix_Access(const char *path, int amode)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of a mkdir of a Unix file system
//
   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      return Xunix.Access(  path, amode);

// Return the results of our version of access()
//
   return Xroot.Access(myPath, amode);
}

/******************************************************************************/
/*                     X r d P o s i x _ C l o s e d i r                      */
/******************************************************************************/

int XrdPosix_Closedir(DIR *dirp)
{
   return (Xroot.isXrootdDir(dirp) ? Xroot.Closedir(dirp)
                                   : Xunix.Closedir(dirp));
}

/******************************************************************************/
/*                        X r d P o s i x _ F c n t l                         */
/******************************************************************************/
  
int XrdPosix_Fcntl(int fd, int cmd, ...) {return 0;}

/******************************************************************************/
/*                        X r d P o s i x _ L s e e k                         */
/******************************************************************************/
  
off_t XrdPosix_Lseek(int fildes, off_t offset, int whence)
{

// Return the operation of the seek
//
   return (fildes < XrdPosixFD ? Xunix.Lseek(fildes, offset, whence)
                               : Xroot.Lseek(fildes, offset, whence));
}

/******************************************************************************/
/*                    X r d P o s i x _ F d a t a s y n c                     */
/******************************************************************************/
  
int XrdPosix_Fdatasync(int fildes)
{

// Return the result of the sync
//
   return (fildes < XrdPosixFD ? Xunix.Fdatasync(fildes)
                               : Xroot.Fsync(fildes));
}

/******************************************************************************/
/*                        X r d P o s i x _ F s t a t                         */
/******************************************************************************/
  
int XrdPosix_Fstat(int fildes, struct stat *buf)
{

// Return result of the close
//
   return (fildes < XrdPosixFD
#ifdef __linux__
          ? Xunix.Fstat64(_STAT_VER, fildes, (struct stat64 *)buf)
#else
          ? Xunix.Fstat64(           fildes, (struct stat64 *)buf)
#endif
          : Xroot.Fstat(fildes, buf));
}

/******************************************************************************/
/*                        X r d P o s i x _ F s y n c                         */
/******************************************************************************/
  
int XrdPosix_Fsync(int fildes)
{

// Return the result of the sync
//
   return (fildes < XrdPosixFD ? Xunix.Fsync(fildes)
                               : Xroot.Fsync(fildes));
}

/******************************************************************************/
/*                        X r d P o s i x _ L s t a t                         */
/******************************************************************************/
  
int XrdPosix_Lstat(const char *path, struct stat *buf)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of an open of a Unix file
//
   return (!(myPath = XrootPath.URL(path, buff, sizeof(buff)))
#ifdef __linux__
          ? Xunix.Lstat64(_STAT_VER, path, (struct stat64 *)buf)
#else
          ? Xunix.Lstat64(           path, (struct stat64 *)buf)
#endif
          : Xroot.Stat(myPath, buf));
}
  
/******************************************************************************/
/*                        X r d P o s i x _ M k d i r                         */
/******************************************************************************/

int XrdPosix_Mkdir(const char *path, mode_t mode)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of a mkdir of a Unix file system
//
   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      return Xunix.Mkdir(path, mode);

// Return the results of an mkdir of an xrootd file system
//
   return Xroot.Mkdir(myPath, mode);
}

/******************************************************************************/
/*                         X r d P o s i x _ O p e n                          */
/******************************************************************************/
  
int XrdPosix_Open(const char *path, int oflag, ...)
{
   char *myPath, buff[2048];
   va_list ap;
   int mode;

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of an open of a Unix file
//
   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      {if (!(oflag & O_CREAT)) return Xunix.Open(path, oflag);
       va_start(ap, oflag);
       mode = va_arg(ap, int);
       va_end(ap);
       return Xunix.Open(path, oflag, (mode_t)mode);
      }

// Return the results of an open of an xrootd file
//
   if (!(oflag & O_CREAT)) return Xroot.Open(myPath, oflag);
   va_start(ap, oflag);
   mode = va_arg(ap, int);
   va_end(ap);
   return Xroot.Open(myPath, oflag, (mode_t)mode);
}

/******************************************************************************/
/*                       X r d P o s i x _ O p e n d i r                      */
/******************************************************************************/

DIR* XrdPosix_Opendir(const char *path)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return 0;}
   
// Unix opendir
//
   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      return Xunix.Opendir(path);

// Xrootd opendir
//
   return Xroot.Opendir(myPath);
}


/******************************************************************************/
/*                         X r d P o s i x _ R e a d                          */
/******************************************************************************/
  
ssize_t XrdPosix_Read(int fildes, void *buf, size_t nbyte)
{

// Return the results of the read
//
   return (fildes < XrdPosixFD ? Xunix.Read(fildes, buf, nbyte)
                               : Xroot.Read(fildes, buf, nbyte));
}

/******************************************************************************/
/*                        X r d P o s i x _ P r e a d                         */
/******************************************************************************/
  
ssize_t XrdPosix_Pread(int fildes, void *buf, size_t nbyte, off_t offset)
{

// Return the results of the read
//
   return (fildes < XrdPosixFD ? Xunix.Pread(fildes, buf, nbyte, offset)
                               : Xroot.Pread(fildes, buf, nbyte, offset));
}
 
/******************************************************************************/
/*                        X r d P o s i x _ R e a d v                         */
/******************************************************************************/
  
ssize_t XrdPosix_Readv(int fildes, const struct iovec *iov, int iovcnt)
{

// Return results of the readv
//
   return (fildes < XrdPosixFD ? Xunix.Readv(fildes, iov, iovcnt)
                               : Xroot.Readv(fildes, iov, iovcnt));
}

/******************************************************************************/
/*                      X r d P o s i x _ R e a d d i r                       */
/******************************************************************************/


// On some platforms both 32- and 64-bit versions are callable. so do the same
//
struct dirent   * XrdPosix_Readdir  (DIR *dirp)
{
// Return result of readdir
//
   return (Xroot.isXrootdDir(dirp) ? Xroot.Readdir(dirp)
                                   : Xunix.Readdir(dirp));
}

struct dirent64 * XrdPosix_Readdir64(DIR *dirp)
{
// Return result of readdir
//
   return (Xroot.isXrootdDir(dirp) ? Xroot.Readdir64(dirp)
                                   : Xunix.Readdir64(dirp));
}

/******************************************************************************/
/*                    X r d P o s i x _ R e a d d i r _ r                     */
/******************************************************************************/

int XrdPosix_Readdir_r(DIR *dirp, struct dirent *entry, struct dirent **result)
{
// Return result of readdir
//
   return (Xroot.isXrootdDir(dirp) ? Xroot.Readdir_r(dirp,entry,result)
                                   : Xunix.Readdir_r(dirp,entry,result));
}

int XrdPosix_Readdir64_r(DIR *dirp, struct dirent64 *entry, struct dirent64 **result)
{
// Return result of readdir
//
   return (Xroot.isXrootdDir(dirp) ? Xroot.Readdir64_r(dirp,entry,result)
                                   : Xunix.Readdir64_r(dirp,entry,result));
}

/******************************************************************************/
/*                       X r d P o s i x _ R e n a m e                        */
/******************************************************************************/

int XrdPosix_Rename(const char *oldpath, const char *newpath)
{
   char *oldPath, buffold[2048], *newPath, buffnew[2048];

// Make sure a path was passed
//
   if (!oldpath || !newpath) {errno = EFAULT; return -1;}

// Return the results of a mkdir of a Unix file system
//
   if (!(oldPath = XrootPath.URL(oldpath, buffold, sizeof(buffold)))
   ||  !(newPath = XrootPath.URL(newpath, buffnew, sizeof(buffnew))))
      return Xunix.Rename(oldpath, newpath);

// Return the results of an mkdir of an xrootd file system
//
   return Xroot.Rename(oldPath, newPath);
}

/******************************************************************************/
/*                    X r d P o s i x _ R e w i n d d i r                     */
/******************************************************************************/

void XrdPosix_Rewinddir(DIR *dirp)
{
// Return result of rewind
//
   return (Xroot.isXrootdDir(dirp) ? Xroot.Rewinddir(dirp)
                                   : Xunix.Rewinddir(dirp));
}

/******************************************************************************/
/*                        X r d P o s i x _ R m d i r                         */
/******************************************************************************/

int XrdPosix_Rmdir(const char *path)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of a mkdir of a Unix file system
//
   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      return Xunix.Rmdir(path);

// Return the results of an mkdir of an xrootd file system
//
   return Xroot.Rmdir(myPath);
}

/******************************************************************************/
/*                      X r d P o s i x _ S e e k d i r                       */
/******************************************************************************/

void XrdPosix_Seekdir(DIR *dirp, long loc)
{
// Call seekdir
//
   (Xroot.isXrootdDir(dirp) ? Xroot.Seekdir(dirp, loc)
                            : Xunix.Seekdir(dirp, loc));
}

/******************************************************************************/
/*                         X r d P o s i x _ S t a t                          */
/******************************************************************************/
  
int XrdPosix_Stat(const char *path, struct stat *buf)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of an open of a Unix file
//
   return (!(myPath = XrootPath.URL(path, buff, sizeof(buff)))
#ifdef __linux__
          ? Xunix.Stat64(_STAT_VER, path, (struct stat64 *)buf)
#else
          ? Xunix.Stat64(           path, (struct stat64 *)buf)
#endif
          : Xroot.Stat(myPath, buf));
}

/******************************************************************************/
/*                      X r d P o s i x _ T e l l d i r                       */
/******************************************************************************/

long XrdPosix_Telldir(DIR *dirp)
{
// Return result of telldir
//
   return (Xroot.isXrootdDir(dirp) ? Xroot.Telldir(dirp)
                                   : Xunix.Telldir(dirp));
}


/******************************************************************************/
/*                      X r d P o s i x _ U n l i n k                         */
/******************************************************************************/

int XrdPosix_Unlink(const char *path)
{   
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the result of a unlink of a Unix file
//
   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      return Xunix.Unlink(path);

// Return the results of an unlink of an xrootd file
//
   return Xroot.Unlink(myPath);
}

/******************************************************************************/
/*                        X r d P o s i x _ W r i t e                         */
/******************************************************************************/
  
ssize_t XrdPosix_Write(int fildes, const void *buf, size_t nbyte)
{

// Return the results of the write
//
   return (fildes < XrdPosixFD ? Xunix.Write(fildes, buf, nbyte)
                               : Xroot.Write(fildes, buf, nbyte));
}

/******************************************************************************/
/*                       X r d P o s i x _ P w r i t e                        */
/******************************************************************************/
  
ssize_t XrdPosix_Pwrite(int fildes, const void *buf, size_t nbyte, off_t offset)
{

// Return the results of the write
//
   return (fildes < XrdPosixFD ? Xunix.Pwrite(fildes, buf, nbyte, offset)
                               : Xroot.Pwrite(fildes, buf, nbyte, offset));
}
 
/******************************************************************************/
/*                       X r d P o s i x _ W r i t e v                        */
/******************************************************************************/
  
ssize_t XrdPosix_Writev(int fildes, const struct iovec *iov, int iovcnt)
{

// Return results of the writev
//
   return (fildes < XrdPosixFD ? Xunix.Writev(fildes, iov, iovcnt)
                               : Xroot.Writev(fildes, iov, iovcnt));
}

/******************************************************************************/
/*                     X r d P o s i x _ i s M y P a t h                      */
/******************************************************************************/
  
int XrdPosix_isMyPath(const char *path)
{
    return (0 != XrootPath.URL(path, 0, 0));
}
