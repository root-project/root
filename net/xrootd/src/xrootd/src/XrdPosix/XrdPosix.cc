/******************************************************************************/
/*                                                                            */
/*                           X r d P o s i x . c c                            */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdPosixCVSID = "$Id$";

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/uio.h>

#include "XrdSys/XrdSysHeaders.hh"
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
      else if (cwdPlen <= MAXPATHLEN)
           {char buff[MAXPATHLEN+8];
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
   while(*(path+1) == '/') path++;
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
  
       XrdPosixXrootd    Xroot;

       XrdPosixXrootPath XrootPath;
  
extern XrdPosixLinkage   Xunix;

/******************************************************************************/
/*                       X r d P o s i x _ A c c e s s                        */
/******************************************************************************/
  
extern "C"
{
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
}

/******************************************************************************/
/*                          X r d P o s i x _ A c l                           */
/******************************************************************************/

// This is a required addition for Solaris 10+ systems

extern "C"
{
int XrdPosix_Acl(const char *path, int cmd, int nentries, void *aclbufp)
{
   return (XrootPath.URL(path, 0, 0)
        ? Xunix.Acl("/tmp", cmd,nentries,aclbufp)
        : Xunix.Acl(path,   cmd,nentries,aclbufp));
}
}
  
/******************************************************************************/
/*                        X r d P o s i x _ C h d i r                         */
/******************************************************************************/

extern "C"
{
int XrdPosix_Chdir(const char *path)
{
   int rc;

// Set the working directory if the actual chdir succeeded
//
   if (!(rc = Xunix.Chdir(path))) XrootPath.CWD(path);
   return rc;
}
}
  
/******************************************************************************/
/*                        X r d P o s i x _ C l o s e                         */
/******************************************************************************/

extern "C"
{
int XrdPosix_Close(int fildes)
{

// Return result of the close
//
   return (Xroot.myFD(fildes) ? Xroot.Close(fildes) : Xunix.Close(fildes));
}
}

/******************************************************************************/
/*                     X r d P o s i x _ C l o s e d i r                      */
/******************************************************************************/

extern "C"
{
int XrdPosix_Closedir(DIR *dirp)
{

   return (Xroot.isXrootdDir(dirp) ? Xroot.Closedir(dirp)
                                   : Xunix.Closedir(dirp));
}
}

/******************************************************************************/
/*                        X r d P o s i x _ C r e a t                         */
/******************************************************************************/
  
extern "C"
{
int     XrdPosix_Creat(const char *path, mode_t mode)
{
   extern int XrdPosix_Open(const char *path, int oflag, ...);

   return XrdPosix_Open(path, O_WRONLY | O_CREAT | O_TRUNC, mode);
}
}
  
/******************************************************************************/
/*                       X r d P o s i x _ F c l o s e                        */
/******************************************************************************/

extern "C"
{
int XrdPosix_Fclose(FILE *stream)
{
   int nullfd = fileno(stream);

// Close the associated file
//
   if (Xroot.myFD(nullfd)) Xroot.Close(nullfd, 1);

// Now close the stream
//
   return Xunix.Fclose(stream);
}
}

/******************************************************************************/
/*                        X r d P o s i x _ F c n t l                         */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Fcntl(int fd, int cmd, ...)
{
   va_list ap;
   void *theArg;

   if (Xroot.myFD(fd)) return 0;
   va_start(ap, cmd);
   theArg = va_arg(ap, void *);
   va_end(ap);
   return Xunix.Fcntl64(fd, cmd, theArg);
}
}

/******************************************************************************/
/*                    X r d P o s i x _ F d a t a s y n c                     */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Fdatasync(int fildes)
{

// Return the result of the sync
//
   return (Xroot.myFD(fildes) ? Xroot.Fsync(fildes)
                              : Xunix.Fsync(fildes));
}
}

/******************************************************************************/
/*                    X r d P o s i x _ F g e t x a t t r                     */
/******************************************************************************/
  
#ifdef __linux__
extern "C"
{
long long XrdPosix_Fgetxattr (int fd, const char *name, void *value, 
                              unsigned long long size)
{
   if (Xroot.myFD(fd)) {errno = ENOTSUP; return -1;}
   return Xunix.Fgetxattr(fd, name, value, size);
}
}
#endif

/******************************************************************************/
/*                       X r d P o s i x _ F f l u s h                        */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Fflush(FILE *stream)
{

// Return the result of the fseek
//
   if (!stream || !Xroot.myFD(fileno(stream)))
      return Xunix.Fflush(stream);

   return Xroot.Fsync(fileno(stream));
}
}
  
/******************************************************************************/
/*                        X r d P o s i x _ F o p e n                         */
/******************************************************************************/

#define ISMODE(x) !strcmp(mode, x)
  
extern "C"
{
FILE *XrdPosix_Fopen(const char *path, const char *mode)
{
   char *myPath, buff[2048];
   int erc, fd, omode;
   FILE *stream;

// Transfer to unix if this is not our path
//
   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      return Xunix.Fopen64(path, mode);

// Translate the mode flags
//
        if (ISMODE("r")  || ISMODE("rb"))                   omode = O_RDONLY;
   else if (ISMODE("w")  || ISMODE("wb"))                   omode = O_WRONLY
                                                        | O_CREAT | O_TRUNC;
   else if (ISMODE("a")  || ISMODE("ab"))                   omode = O_APPEND;
   else if (ISMODE("r+") || ISMODE("rb+") || ISMODE("r+b")) omode = O_RDWR;
   else if (ISMODE("w+") || ISMODE("wb+") || ISMODE("w+b")) omode = O_RDWR
                                                        | O_CREAT | O_TRUNC;
   else if (ISMODE("a+") || ISMODE("ab+") || ISMODE("a+b")) omode = O_APPEND;
   else {errno = EINVAL; return 0;}

// Now open the file
//
   if ((fd = Xroot.Open(myPath, omode | XrdPosixXrootd::isStream , 0)) < 0)
      return 0;

// First obtain a free stream
//
   if (!(stream = fdopen(fd, mode))) 
      {erc = errno; Xroot.Close(fd); errno = erc;}

// All done
//
   return stream;
}
}

/******************************************************************************/
/*                        X r d P o s i x _ F r e a d                         */
/******************************************************************************/
  
extern "C"
{
size_t XrdPosix_Fread(void *ptr, size_t size, size_t nitems, FILE *stream)
{
   ssize_t bytes;
   size_t rc = 0;
   int fd = fileno(stream);

   if (!Xroot.myFD(fd)) return Xunix.Fread(ptr, size, nitems, stream);

   bytes = Xroot.Read(fd, ptr, size*nitems);

// Get the right return code. Note that we cannot emulate the flags in sunx86
//
        if (bytes > 0 && size) rc = bytes/size;
#ifndef SUNX86
#if defined(__linux__)
   else if (bytes < 0) stream->_flags |= _IO_ERR_SEEN;
   else                stream->_flags |= _IO_EOF_SEEN;
#elif defined(__macos__)
   else if (bytes < 0) stream->_flags |= __SEOF;
   else                stream->_flags |= __SERR;
#else
   else if (bytes < 0) stream->_flag  |= _IOERR;
   else                stream->_flag  |= _IOEOF;
#endif
#endif

   return rc;
}
}
  
/******************************************************************************/
/*                        X r d P o s i x _ F s e e k                         */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Fseek(FILE *stream, long offset, int whence)
{

// Return the result of the fseek
//
   if (!Xroot.myFD(fileno(stream)))
      return Xunix.Fseek( stream, offset, whence);

   return (Xroot.Lseek(fileno(stream), offset, whence) < 0 ? -1 : 0);
}
}

/******************************************************************************/
/*                       X r d P o s i x _ F s e e k o                        */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Fseeko(FILE *stream, long long offset, int whence)
{

// Return the result of the fseek
//
   if (!Xroot.myFD(fileno(stream)))
      return Xunix.Fseeko64(stream, offset, whence);

   return (Xroot.Lseek(fileno(stream), offset, whence) < 0 ? -1 : 0);
}
}

/******************************************************************************/
/*                        X r d P o s i x _ F s t a t                         */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Fstat(int fildes, struct stat *buf)
{

// Return result of the close
//
   return (Xroot.myFD(fildes)
          ? Xroot.Fstat(fildes, buf)
#ifdef __linux__
          : Xunix.Fstat64(_STAT_VER, fildes, (struct stat64 *)buf));
#else
          : Xunix.Fstat64(           fildes, (struct stat64 *)buf));
#endif
}

#ifdef __linux__
int XrdPosix_FstatV(int ver, int fildes, struct stat *buf)
{
   return (Xroot.myFD(fildes)
          ? Xroot.Fstat(fildes, buf)
          : Xunix.Fstat64(ver, fildes, (struct stat64 *)buf));
}
#endif
}

/******************************************************************************/
/*                        X r d P o s i x _ F s y n c                         */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Fsync(int fildes)
{

// Return the result of the sync
//
   return (Xroot.myFD(fildes) ? Xroot.Fsync(fildes)
                              : Xunix.Fsync(fildes));
}
}
  
/******************************************************************************/
/*                        X r d P o s i x _ F t e l l                         */
/******************************************************************************/
  
extern "C"
{
long XrdPosix_Ftell(FILE *stream)
{

// Return the result of the tell
//
   if (!Xroot.myFD(fileno(stream))) return Xunix.Ftell(stream);

   return static_cast<long>(Xroot.Lseek(fileno(stream), 0, SEEK_CUR));
}
}
  
/******************************************************************************/
/*                       X r d P o s i x _ F t e l l o                        */
/******************************************************************************/
  
extern "C"
{
long long XrdPosix_Ftello(FILE *stream)
{

// Return the result of the tell
//
   if (!Xroot.myFD(fileno(stream))) return Xunix.Ftello64(stream);

   return Xroot.Lseek(fileno(stream), 0, SEEK_CUR);
}
}
  
/******************************************************************************/
/*                    X r d P o s i x _ F t r u n c a t e                     */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Ftruncate(int fildes, long long offset)
{

// Return the result of the ftruncate
//
   return (Xroot.myFD(fildes) ? Xroot.Ftruncate  (fildes, offset)
                              : Xunix.Ftruncate64(fildes, offset));
}
}

/******************************************************************************/
/*                       X r d P o s i x _ F w r i t e                        */
/******************************************************************************/
  
extern "C"
{
size_t XrdPosix_Fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream)
{
   size_t bytes, rc = 0;
   int fd = fileno(stream);

   if (!Xroot.myFD(fd)) return Xunix.Fwrite(ptr, size, nitems, stream);

   bytes = Xroot.Write(fd, ptr, size*nitems);

// Get the right return code. Note that we cannot emulate the flags in sunx86
//
   if (bytes > 0 && size) rc = bytes/size;
#ifndef SUNX86
#if defined(__linux__)
      else stream->_flags |= _IO_ERR_SEEN;
#elif defined(__macos__)
      else stream->_flags |= __SERR;
#else
      else stream->_flag  |= _IOERR;
#endif
#endif

   return rc;
}
}
  
/******************************************************************************/
/*                     X r d P o s i x _ G e t x a t t r                      */
/******************************************************************************/
  
#ifdef __linux__
extern "C"
{
long long XrdPosix_Getxattr (const char *path, const char *name, void *value, 
                             unsigned long long size)
{
   char *myPath, buff[2048];

   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      return Xunix.Getxattr(path, name, value, size);

   return Xroot.Getxattr(myPath, name, value, size);
}
}
#endif

/******************************************************************************/
/*                    X r d P o s i x _ L g e t x a t t r                     */
/******************************************************************************/
  
#ifdef __linux__
extern "C"
{
long long XrdPosix_Lgetxattr (const char *path, const char *name, void *value, 
                              unsigned long long size)
{
   if (XrootPath.URL(path, 0, 0)) {errno = ENOTSUP; return -1;}
   return Xunix.Lgetxattr(path, name, value, size);
}
}
#endif

/******************************************************************************/
/*                        X r d P o s i x _ L s e e k                         */
/******************************************************************************/
  
extern "C"
{
long long XrdPosix_Lseek(int fildes, long long offset, int whence)
{

// Return the operation of the seek
//
   return (Xroot.myFD(fildes) ? Xroot.Lseek  (fildes, offset, whence)
                              : Xunix.Lseek64(fildes, offset, whence));
}
}

/******************************************************************************/
/*                        X r d P o s i x _ L s t a t                         */
/******************************************************************************/
  
extern "C"
{
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
}
  
/******************************************************************************/
/*                        X r d P o s i x _ M k d i r                         */
/******************************************************************************/

extern "C"
{
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
}

/******************************************************************************/
/*                         X r d P o s i x _ O p e n                          */
/******************************************************************************/
  
extern "C"
{
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
      {if (!(oflag & O_CREAT)) return Xunix.Open64(path, oflag);
       va_start(ap, oflag);
       mode = va_arg(ap, int);
       va_end(ap);
       return Xunix.Open64(path, oflag, (mode_t)mode);
      }

// Return the results of an open of an xrootd file
//
   if (!(oflag & O_CREAT)) return Xroot.Open(myPath, oflag);
   va_start(ap, oflag);
   mode = va_arg(ap, int);
   va_end(ap);
   return Xroot.Open(myPath, oflag, (mode_t)mode);
}
}

/******************************************************************************/
/*                       X r d P o s i x _ O p e n d i r                      */
/******************************************************************************/

extern "C"
{
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
}

/******************************************************************************/
/*                     X r d P o s i x _ P a t h c o n f                      */
/******************************************************************************/
  
// This is a required addition for Solaris 10+ systems

extern "C"
{
long XrdPosix_Pathconf(const char *path, int name)
{
   return (XrootPath.URL(path, 0, 0) ? Xunix.Pathconf("/tmp", name)
                                     : Xunix.Pathconf(path,   name));
}
}

/******************************************************************************/
/*                        X r d P o s i x _ P r e a d                         */
/******************************************************************************/
  
extern "C"
{
long long XrdPosix_Pread(int fildes, void *buf, unsigned long long nbyte,
                         long long offset)
{

// Return the results of the read
//
   return (Xroot.myFD(fildes) ? Xroot.Pread  (fildes, buf, nbyte, offset)
                              : Xunix.Pread64(fildes, buf, nbyte, offset));
}
}

/******************************************************************************/
/*                       X r d P o s i x _ P w r i t e                        */
/******************************************************************************/
  
extern "C"
{
long long XrdPosix_Pwrite(int fildes, const void *buf, unsigned long long nbyte,
                          long long offset)
{

// Return the results of the write
//
   return (Xroot.myFD(fildes) ? Xroot.Pwrite  (fildes, buf, nbyte, offset)
                              : Xunix.Pwrite64(fildes, buf, nbyte, offset));
}
}

/******************************************************************************/
/*                         X r d P o s i x _ R e a d                          */
/******************************************************************************/
  
extern "C"
{
long long XrdPosix_Read(int fildes, void *buf, unsigned long long nbyte)
{

// Return the results of the read
//
   return (Xroot.myFD(fildes) ? Xroot.Read(fildes, buf, nbyte)
                              : Xunix.Read(fildes, buf, nbyte));
}
}
 
/******************************************************************************/
/*                        X r d P o s i x _ R e a d v                         */
/******************************************************************************/
  
extern "C"
{
long long XrdPosix_Readv(int fildes, const struct iovec *iov, int iovcnt)
{

// Return results of the readv
//
   return (Xroot.myFD(fildes) ? Xroot.Readv(fildes, iov, iovcnt)
                              : Xunix.Readv(fildes, iov, iovcnt));
}
}

/******************************************************************************/
/*                      X r d P o s i x _ R e a d d i r                       */
/******************************************************************************/

extern "C"
{
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
}

/******************************************************************************/
/*                    X r d P o s i x _ R e a d d i r _ r                     */
/******************************************************************************/

extern "C"
{
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
}

/******************************************************************************/
/*                       X r d P o s i x _ R e n a m e                        */
/******************************************************************************/

extern "C"
{
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
}

/******************************************************************************/
/*                    X r d P o s i x _ R e w i n d d i r                     */
/******************************************************************************/

extern "C"
{
void XrdPosix_Rewinddir(DIR *dirp)
{

// Return result of rewind
//
   return (Xroot.isXrootdDir(dirp) ? Xroot.Rewinddir(dirp)
                                   : Xunix.Rewinddir(dirp));
}
}

/******************************************************************************/
/*                        X r d P o s i x _ R m d i r                         */
/******************************************************************************/

extern "C"
{
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
}

/******************************************************************************/
/*                      X r d P o s i x _ S e e k d i r                       */
/******************************************************************************/

extern "C"
{
void XrdPosix_Seekdir(DIR *dirp, long loc)
{

// Call seekdir
//
   (Xroot.isXrootdDir(dirp) ? Xroot.Seekdir(dirp, loc)
                            : Xunix.Seekdir(dirp, loc));
}
}

/******************************************************************************/
/*                         X r d P o s i x _ S t a t                          */
/******************************************************************************/
  
extern "C"
{
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
}
  
/******************************************************************************/
/*                       X r d P o s i x _ S t a t f s                        */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Statfs(const char *path, struct statfs *buf)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of an open of a Unix file
//
   return ((myPath = XrootPath.URL(path, buff, sizeof(buff)))
          ? Xroot.Statfs(myPath, buf) 
          : Xunix.Statfs64(path, (struct statfs64 *)buf));
}
}
  
/******************************************************************************/
/*                      X r d P o s i x _ S t a t v f s                       */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Statvfs(const char *path, struct statvfs *buf)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of an open of a Unix file
//
   return ((myPath = XrootPath.URL(path, buff, sizeof(buff)))
          ? Xroot.Statvfs(myPath, buf) 
          : Xunix.Statvfs64(path, (struct statvfs64 *)buf));
}
}

/******************************************************************************/
/*                      X r d P o s i x _ T e l l d i r                       */
/******************************************************************************/

extern "C"
{
long XrdPosix_Telldir(DIR *dirp)
{

// Return result of telldir
//
   return (Xroot.isXrootdDir(dirp) ? Xroot.Telldir(dirp)
                                   : Xunix.Telldir(dirp));
}
}

/******************************************************************************/
/*                     X r d P o s i x _ T r u n c a t e                      */
/******************************************************************************/
  
extern "C"
{
int XrdPosix_Truncate(const char *path, long long offset)
{
   char *myPath, buff[2048];

// Make sure a path was passed
//
   if (!path) {errno = EFAULT; return -1;}

// Return the results of a truncate of a Unix file system
//
   if (!(myPath = XrootPath.URL(path, buff, sizeof(buff))))
      return Xunix.Truncate64(path, offset);

// Return the results of an truncate of an xrootd file system
//
   return Xroot.Truncate(myPath, offset);
}
}
  
/******************************************************************************/
/*                      X r d P o s i x _ U n l i n k                         */
/******************************************************************************/

extern "C"
{
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
}

/******************************************************************************/
/*                        X r d P o s i x _ W r i t e                         */
/******************************************************************************/
  
extern "C"
{
long long XrdPosix_Write(int fildes, const void *buf, unsigned long long nbyte)
{

// Return the results of the write
//
   return (Xroot.myFD(fildes) ? Xroot.Write(fildes, buf, nbyte)
                              : Xunix.Write(fildes, buf, nbyte));
}
}
 
/******************************************************************************/
/*                       X r d P o s i x _ W r i t e v                        */
/******************************************************************************/
  
extern "C"
{
long long XrdPosix_Writev(int fildes, const struct iovec *iov, int iovcnt)
{

// Return results of the writev
//
   return (Xroot.myFD(fildes) ? Xroot.Writev(fildes, iov, iovcnt)
                              : Xunix.Writev(fildes, iov, iovcnt));
}
}

/******************************************************************************/
/*                     X r d P o s i x _ i s M y P a t h                      */
/******************************************************************************/
  
int XrdPosix_isMyPath(const char *path)
{
    return (0 != XrootPath.URL(path, 0, 0));
}

/******************************************************************************/
/*                          X r d P o s i x _ U R L                           */
/******************************************************************************/
  
char *XrdPosix_URL(const char *path, char *buff, int blen)
{
   return XrootPath.URL(path, buff, blen);
}
