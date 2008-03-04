/******************************************************************************/
/*                                                                            */
/*                     X r d P o s i x X r o o t d . c c                      */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
const char *XrdPosixXrootdCVSID = "$Id$";

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/param.h>
#include <sys/resource.h>
#include <sys/uio.h>

#include "XrdClient/XrdClient.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientAdmin.hh"
#include "XrdClient/XrdClientUrlSet.hh"
#include "XrdClient/XrdClientVector.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdPosixXrootd.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
/******************************************************************************/
/*                X r d P o s i x A d m i n N e w   C l a s s                 */
/******************************************************************************/
  
class XrdPosixAdminNew
{
public:

XrdClientAdmin Admin;

int            Fault();

int            isOK() {return eNum == 0;}

int            lastError()
                   {return XrdPosixXrootd::mapError(Admin.LastServerError()->errnum);}

int            Result() {if (eNum) {errno = eNum; return -1;}
                         return 0;
                        }

      XrdPosixAdminNew(const char *path);
     ~XrdPosixAdminNew() {}

private:

int            eNum;
};

/******************************************************************************/
/*                     X r d P o s i x D i r   C l a s s                      */
/******************************************************************************/

class XrdPosixDir {

public:
  XrdPosixDir(int dirno, const char *path);
 ~XrdPosixDir();

  void             Lock()   { myMutex.Lock(); }
  void             UnLock() { myMutex.UnLock(); }
  int              dirNo()  { return fdirno; }
  long             getEntries() { return fentries.GetSize(); }
  long             getOffset() { return fentry; }
  void             setOffset(long offset) { fentry = offset; }
  dirent64        *nextEntry(dirent64 *dp=0);
  void             rewind() { fentry = -1; fentries.Clear();}
  int              Status() { return eNum;}

  
private:
  XrdSysMutex     myMutex;
  XrdClientAdmin  XAdmin;
  dirent64       *myDirent;
  static int      maxname;
  int             fdirno;
  char           *fpath;
  vecString       fentries;
  long            fentry;
  int             eNum;
};
  
/******************************************************************************/
/*                    X r d P o s i x F i l e   C l a s s                     */
/******************************************************************************/
  
class XrdPosixFile
{
public:

XrdClient *XClient;

long long  Offset() {return currOffset;}

long long  addOffset(long long offs, int updtSz=0)
                    {currOffset += offs;
                     if (updtSz && currOffset > stat.size) stat.size=currOffset;
                     return currOffset;
                    }

long long  setOffset(long long offs)
                    {currOffset = offs;
                     return currOffset;
                    }

void         Lock() {myMutex.Lock();}
void       UnLock() {myMutex.UnLock();}

int               FD;

XrdClientStatInfo stat;

           XrdPosixFile(int fd, const char *path);
          ~XrdPosixFile();

private:

XrdSysMutex myMutex;
long long   currOffset;
};


typedef XrdClientVector<XrdOucString> vecString;
typedef XrdClientVector<bool> vecBool;

/******************************************************************************/
/*                         L o c a l   D e f i n e s                          */
/******************************************************************************/
  
#define Scuttle(fp, rc) fp->UnLock(); errno = rc; return -1

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

int            XrdPosixDir::maxname = (pathconf("./",_PC_NAME_MAX) > 0 ?
                                       pathconf("./",_PC_NAME_MAX) : 255);
XrdSysMutex    XrdPosixXrootd::myMutex;
XrdPosixFile **XrdPosixXrootd::myFiles  =  0;
XrdPosixDir  **XrdPosixXrootd::myDirs   =  0;
int            XrdPosixXrootd::highFD   = -1;
int            XrdPosixXrootd::lastFD   = -1;
int            XrdPosixXrootd::highDir  = -1;
int            XrdPosixXrootd::lastDir  = -1;
long           XrdPosixXrootd::Debug    = -2;
const int      XrdPosixXrootd::FDMask   = 0x00003fff;
const int      XrdPosixXrootd::FDOffs   = 0x00004000;
const int      XrdPosixXrootd::FDLeft   = 0x7fffC000;
  
/******************************************************************************/
/*                X r d P o s i x A d m i n N e w   C l a s s                 */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdPosixAdminNew::XrdPosixAdminNew(const char *path) : Admin(path)
{
   if (!Admin.Connect())
       eNum = XrdPosixXrootd::mapError(Admin.LastServerError()->errnum);
       else eNum = 0;
}

/******************************************************************************/
/*                                 F a u l t                                  */
/******************************************************************************/

int XrdPosixAdminNew::Fault()
{
   char *etext = Admin.LastServerError()->errmsg;
   int RC = XrdPosixXrootd::mapError(Admin.LastServerError()->errnum);

   if (RC != ENOENT && *etext && XrdPosixXrootd::Debug > -2)
      cerr <<"XrdPosix: " <<etext <<endl;
   errno = RC;
   return -1;
}
  
/******************************************************************************/
/*                           X r d P o s i x D i r                            */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdPosixDir::XrdPosixDir(int dirno, const char *path) : XAdmin(path)
{
   if (!XAdmin.Connect())
      eNum = XrdPosixXrootd::mapError(XAdmin.LastServerError()->errnum);
      else eNum = 0;

   fentry = -1;     // indicates that the directory content is not valid
   fentries.Clear();
   fdirno = dirno;

// Get the path of the url 
//
   XrdOucString str(path);
   XrdClientUrlSet url(str);
   XrdOucString dir = url.GetFile();
   fpath = strdup(dir.c_str());

// Allocate a local dirent. Note that we get additional padding because on
// some system the dirent structure does not include the name buffer
//
   if (!(myDirent = (dirent64 *)malloc(sizeof(dirent64) + maxname + 1)))
      eNum = ENOMEM;
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdPosixDir::~XrdPosixDir()
{
  if (fpath)    free(fpath);
  if (myDirent) free(myDirent);
}

/******************************************************************************/
/*                            n e x t E n t r y                               */
/******************************************************************************/

dirent64 *XrdPosixDir::nextEntry(dirent64 *dp)
{
   const char *cp;
   const int dirhdrln = dp->d_name - (char *)dp;
   int reclen;

// Object is already / must be locked at this point
// Read directory if we haven't done that yet
//
   if (fentry<0)
      {if (XAdmin.DirList(fpath,fentries)) fentry = 0;
          else {eNum = XrdPosixXrootd::mapError(XAdmin.LastServerError()->errnum);
                return 0;
               }
      }

// Check if dir is empty or all entries have been read
//
   if ((fentries.GetSize()==0) || (fentry>=fentries.GetSize())) return 0;

// Create a directory entry
//
   if (!dp) dp = myDirent;
   cp = (fentries[fentry]).c_str();
   reclen = strlen(cp);
   if (reclen > maxname) reclen = maxname;
#ifdef __macos__
   dp->d_fileno = fentry;
   dp->d_type   = DT_UNKNOWN;
   dp->d_namlen = reclen;
#else
   dp->d_ino    = fentry;
   dp->d_off    = fentry*maxname;
#endif
   dp->d_reclen = reclen + dirhdrln;
   strncpy(dp->d_name, cp, reclen);
   dp->d_name[reclen] = '\0';
   fentry++;
   return dp;
}

/******************************************************************************/
/*                          X r d P o s i x F i l e                           */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdPosixFile::XrdPosixFile(int fd, const char *path)
             : FD(fd),
               currOffset(0)
{
// Allocate a new client object
//
   if (!(XClient = new XrdClient(path))) stat.size = 0;
}
  
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdPosixFile::~XrdPosixFile()
{
   if (XClient) delete XClient;
}

/******************************************************************************/
/*                         X r d P o s i x X r o o t d                        */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdPosixXrootd::XrdPosixXrootd(int fdnum, int dirnum)
{
   struct rlimit rlim;
   char *cvar;
   long isize;

// Compute size of table
//
   if (!getrlimit(RLIMIT_NOFILE, &rlim)) fdnum = (int)rlim.rlim_cur;
   if (fdnum > 32768) fdnum = 32768;
   isize = fdnum * sizeof(XrdPosixFile *);

// Allocate an initial table of 64 fd-type pointers
//
   if (!(myFiles = (XrdPosixFile **)malloc(isize))) lastFD = -1;
      else {memset((void *)myFiles, 0, isize); lastFD = fdnum;}

// Allocate table for DIR descriptors
//
   if (dirnum > 32768) dirnum = 32768;
   isize = dirnum * sizeof(XrdPosixDir *);
   if (!(myDirs = (XrdPosixDir **)malloc(isize))) lastDir = -1;
   else {
     memset((void *)myDirs, 0, isize);
     lastDir = dirnum;
   }

// Establish debugging level
//
   if ((cvar = getenv("XRDPOSIX_DEBUG")) && *cvar)
      {Debug = atol(cvar); setEnv(NAME_DEBUG, Debug);}

// Establish read ahead size
//
   if ((cvar = getenv("XRDPOSIX_RASZ")) && *cvar)
      {isize = atol(cvar); setEnv(NAME_READAHEADSIZE, isize);}

// Establish cache size
//
   if ((cvar = getenv("XRDPOSIX_RCSZ")) && *cvar)
      {isize = atol(cvar); setEnv(NAME_READCACHESIZE, isize);}
}
 
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdPosixXrootd::~XrdPosixXrootd()
{
   int i;

   if (myFiles)
      {for (i = 0; i <= highFD; i++) if (myFiles[i]) delete myFiles[i];
       free(myFiles); myFiles = 0;
      }

   if (myDirs) {
     for (i = 0; i <= highDir; i++)
       if (myDirs[i]) delete myDirs[i];
     free(myDirs); myDirs = 0;
   }
}
 
/******************************************************************************/
/*                                A c c e s s                                 */
/******************************************************************************/
  
int XrdPosixXrootd::Access(const char *path, int amode)
{
   XrdPosixAdminNew admin(path);
   long st_flags, st_modtime, st_id;
   long long st_size;
   int st_mode, aOK = 1;

// Make sure we connected
//
   if (!admin.isOK()) return admin.Result();

// Extract out path from the url
//
   XrdOucString str(path);
   XrdClientUrlSet url(str);

// Open the file first
//
   if (!admin.Admin.Stat(url.GetFile().c_str(),
                         st_id, st_size, st_flags, st_modtime))
      {errno = admin.lastError();
       return -1;
      }

// Translate the mode bits
//
   st_mode = mapFlags(st_flags);
   if (amode & R_OK && !(st_mode & S_IRUSR)) aOK = 0;
   if (amode & W_OK && !(st_mode & S_IWUSR)) aOK = 0;
   if (amode & X_OK && !(st_mode & S_IXUSR)) aOK = 0;

// All done
//
   if (aOK) return 0;
   errno = EACCES;
   return -1;
}

/******************************************************************************/
/*                                 C l o s e                                  */
/******************************************************************************/

int     XrdPosixXrootd::Close(int fildes)
{
   XrdPosixFile *fp;

// Find the file object. We tell findFP() to leave the global lock on
//
   if (!(fp = findFP(fildes, 1))) return -1;

// Deallocate the file. We have the global lock.
//
   myFiles[fp->FD] = 0;
   fp->UnLock();
   myMutex.UnLock();
   delete fp;
   return 0;
}

/******************************************************************************/
/*                              C l o s e d i r                               */
/******************************************************************************/

int XrdPosixXrootd::Closedir(DIR *dirp)
{
   XrdPosixDir *XrdDirp = findDIR(dirp,1);
   if (!XrdDirp) return -1;

// Deallocate the directory
//
   myDirs[XrdDirp->dirNo()] = 0;
   XrdDirp->UnLock();
   myMutex.UnLock();
   if (XrdDirp) delete XrdDirp;
   return 0;
}
  
/******************************************************************************/
/*                                 L s e e k                                  */
/******************************************************************************/
  
off_t   XrdPosixXrootd::Lseek(int fildes, off_t offset, int whence)
{
   XrdPosixFile *fp;
   long long curroffset;

// Find the file object
//
   if (!(fp = findFP(fildes))) return -1;

// Set the new offset
//
   if (whence == SEEK_SET) curroffset = fp->setOffset(offset);
      else if (whence == SEEK_CUR) curroffset = fp->addOffset(offset);
              else if (whence == SEEK_END)
                      curroffset = fp->setOffset(fp->stat.size+offset);
                      else {Scuttle(fp, EINVAL);}

// All done
//
   fp->UnLock();
   return curroffset;
}

/******************************************************************************/
/*                                 F s t a t                                  */
/******************************************************************************/

int     XrdPosixXrootd::Fstat(int fildes, struct stat *buf)
{
   XrdPosixFile *fp;

// Find the file object
//
   if (!(fp = findFP(fildes))) return -1;

// Return what little we can
//
   initStat(buf);
   buf->st_size   = fp->stat.size;
   buf->st_atime  = buf->st_mtime = buf->st_ctime = fp->stat.modtime;
   buf->st_blocks = buf->st_size/512+1;
   buf->st_ino    = fp->stat.id;
   buf->st_mode   = mapFlags(fp->stat.flags);

// All done
//
   fp->UnLock();
   return 0;
}
  
/******************************************************************************/
/*                                 F s y n c                                  */
/******************************************************************************/
  
int     XrdPosixXrootd::Fsync(int fildes)
{
   XrdPosixFile *fp;

// Find the file object
//
   if (!(fp = findFP(fildes))) return -1;

// Do the sync
//
   if (!fp->XClient->Sync()) return Fault(fp);
   fp->UnLock();
   return 0;
}


/******************************************************************************/
/*                                 M k d i r                                  */
/******************************************************************************/

int XrdPosixXrootd::Mkdir(const char *path, mode_t mode)
{
  XrdPosixAdminNew admin(path);
  int uMode = 0, gMode = 0, oMode = 0;

  if (admin.isOK())
     {XrdOucString str(path);
      XrdClientUrlSet url(str);
      if (mode & S_IRUSR) uMode |= 4;
      if (mode & S_IWUSR) uMode |= 2;
      if (mode & S_IXUSR) uMode |= 1;
      if (mode & S_IRGRP) gMode |= 4;
      if (mode & S_IWGRP) gMode |= 2;
      if (mode & S_IXGRP) gMode |= 1;
      if (mode & S_IROTH) oMode |= 4;
      if (mode & S_IXOTH) oMode |= 1;
      if (admin.Admin.Mkdir(url.GetFile().c_str(), uMode, gMode, oMode))
         return 0;
      return admin.Fault();
     }
  return admin.Result();
}

/******************************************************************************/
/*                                  O p e n                                   */
/******************************************************************************/
  
int     XrdPosixXrootd::Open(const char *path, int oflags, mode_t mode)
{
   XrdPosixFile *fp;
   int retc = 0, fd, XOflags, XMode;

// Allocate a new file descriptor.
//
   myMutex.Lock();
   for (fd = 0; fd < lastFD; fd++) if (!myFiles[fd]) break;
   if (fd > lastFD || !(fp = new XrdPosixFile(fd, path)))
      {errno = EMFILE; myMutex.UnLock(); return -1;}
   myFiles[fd] = fp;
   if (fd > highFD) highFD = fd;
   myMutex.UnLock();

// Translate option bits to the appropraite values. Always 
// make directory path for new file.
//
   XOflags = (oflags & (O_WRONLY | O_RDWR) ? kXR_open_updt : kXR_open_read);
   if (oflags & O_CREAT) {
       XOflags |= (oflags & O_EXCL ? kXR_new : kXR_delete);
       XOflags |= kXR_mkpath | kXR_new;
   }
   else if (oflags & O_TRUNC && XOflags & kXR_open_updt)
              XOflags |= kXR_delete;

// Translate the mode, if need be
//
   XMode = (mode && (oflags & O_CREAT) ? mapMode(mode) : 0);

// Open the file
//
   if (!fp->XClient->Open(XMode, XOflags)
   || (fp->XClient->LastServerResp()->status) != kXR_ok)
      {retc = Fault(fp, 0);
       myMutex.Lock();
       myFiles[fd] = 0;
       delete fp;
       myMutex.UnLock();
       errno = retc;
       return -1;
      }

// Get the file size
//
   fp->XClient->Stat(&fp->stat);

// Return the fd number
//
   return fd | FDOffs;
}

/******************************************************************************/
/*                               O p e n d i r                                */
/******************************************************************************/
  
DIR* XrdPosixXrootd::Opendir(const char *path)
{
   XrdPosixDir *dirp = 0; // Avoid MacOS compiler warning
   int rc, dir;

// Allocate a new directory structure
//
   myMutex.Lock();
   for (dir = 0; dir < lastDir; dir++) if (!myDirs[dir]) break;

   if (dir > lastDir) rc = EMFILE;
      else if (!(dirp = new XrdPosixDir(dir, path))) rc = ENOMEM;
              else rc = dirp->Status();

   if (!rc)
      {myDirs[dir] = dirp;
       if (dir > highDir) highDir = dir;
      }
   myMutex.UnLock();

   if (rc) {if (dirp) {delete dirp; dirp = 0;} errno = rc;}

   return (DIR*)dirp;
}

/******************************************************************************/
/*                                 P r e a d                                  */
/******************************************************************************/
  
ssize_t XrdPosixXrootd::Pread(int fildes, void *buf, size_t nbyte, off_t offset)
{
   XrdPosixFile *fp;
   long long     offs, bytes;
   int           iosz;

// Find the file object
//
   if (!(fp = findFP(fildes))) return -1;

// Make sure the size is not too large
//
   if (nbyte > (size_t)0x7fffffff) {Scuttle(fp,EOVERFLOW);}
      else iosz = static_cast<int>(nbyte);

// Issue the read
//
   offs = static_cast<long long>(offset);
   if ((bytes = fp->XClient->Read(buf, offs, (int)iosz))<0) return Fault(fp);

// All went well
//
   fp->UnLock();
   return (ssize_t)bytes;
}

/******************************************************************************/
/*                                  R e a d                                   */
/******************************************************************************/
  
ssize_t XrdPosixXrootd::Read(int fildes, void *buf, size_t nbyte)
{
   XrdPosixFile *fp;
   long long     bytes;
   int           iosz;

// Find the file object
//
   if (!(fp = findFP(fildes))) return -1;


// Make sure the size is not too large
//
   if (nbyte > (size_t)0x7fffffff) {Scuttle(fp,EOVERFLOW);}
      else iosz = static_cast<int>(nbyte);

// Issue the read
//
   if ((bytes = fp->XClient->Read(buf, fp->Offset(), iosz)) < 0)
      return Fault(fp);

// All went well
//
   fp->addOffset(bytes);
   fp->UnLock();
   return (ssize_t)bytes;
}

/******************************************************************************/
/*                                 R e a d v                                  */
/******************************************************************************/
  
ssize_t XrdPosixXrootd::Readv(int fildes, const struct iovec *iov, int iovcnt)
{
   ssize_t bytes, totbytes = 0;
   int i;

// Return the results of the read for each iov segment
//
   for (i = 0; i < iovcnt; i++)
       {if ((bytes = Read(fildes,(void *)iov[i].iov_base,(size_t)iov[i].iov_len)))
           return -1;
        totbytes += bytes;
       }

// All done
//
   return totbytes;
}

/******************************************************************************/
/*                                R e a d d i r                               */
/******************************************************************************/

struct dirent* XrdPosixXrootd::Readdir(DIR *dirp)
{
   dirent64 *dp64;
   dirent   *dp32; // Could be the same as dp64

   if (!(dp64 = Readdir64(dirp))) return 0;

   dp32 = (struct dirent *)dp64;
   if (dp32->d_name != dp64->d_name)
      {dp32->d_ino    = dp64->d_ino;
#if !defined(__macos__)
       dp32->d_off    = dp64->d_off;
#endif
       dp32->d_reclen = dp64->d_reclen;
       strcpy(dp32->d_name, dp64->d_name);
      }
   return dp32;
}

struct dirent64* XrdPosixXrootd::Readdir64(DIR *dirp)
{
   XrdPosixDir *XrdDirp = findDIR(dirp);
   dirent64 *dp;
   int rc;

// Returns the next directory entry
//
   if (!XrdDirp) return 0;

   if (!(dp = XrdDirp->nextEntry())) rc = XrdDirp->Status();
      else rc = 0;


   XrdDirp->UnLock();
   if (rc) errno = rc;
   return dp;
}

/******************************************************************************/
/*                              R e a d d i r _ r                             */
/******************************************************************************/

int XrdPosixXrootd::Readdir_r(DIR *dirp,   struct dirent    *entry,
                                           struct dirent   **result)
{
   dirent64 *dp64;
   int       rc;

   if ((rc = Readdir64_r(dirp, 0, &dp64)) <= 0) {*result = 0; return rc;}

   entry->d_ino    = dp64->d_ino;
#if !defined(__macos__)
   entry->d_off    = dp64->d_off;
#endif
   entry->d_reclen = dp64->d_reclen;
   strcpy(entry->d_name, dp64->d_name);
   *result = entry;
   return rc;
}

int XrdPosixXrootd::Readdir64_r(DIR *dirp, struct dirent64  *entry,
                                           struct dirent64 **result)
{
   XrdPosixDir *XrdDirp = findDIR(dirp);
   int rc;

// Thread safe verison of readdir
//
   if (!XrdDirp) {errno = EBADF; return -1;}

   if (!(*result = XrdDirp->nextEntry(entry))) rc = XrdDirp->Status();
      else rc = 0;

   XrdDirp->UnLock();
   return rc;
}

/******************************************************************************/
/*                                R e n a m e                                 */
/******************************************************************************/

int XrdPosixXrootd::Rename(const char *oldpath, const char *newpath)
{
  XrdPosixAdminNew admin(oldpath);

  if (admin.isOK())
     {XrdOucString    oldstr(oldpath);
      XrdClientUrlSet oldurl(oldstr);
      XrdOucString    newstr(newpath);
      XrdClientUrlSet newurl(newstr);
      if (admin.Admin.Mv(oldurl.GetFile().c_str(),
                         newurl.GetFile().c_str())) return 0;
      return admin.Fault();
     }
  return admin.Result();
}


/******************************************************************************/
/*                            R e w i n d d i r                               */
/******************************************************************************/

void XrdPosixXrootd::Rewinddir(DIR *dirp)
{
// Updates and rewinds directory
//
   XrdPosixDir *XrdDirp = findDIR(dirp);
   if (!XrdDirp) return;

   XrdDirp->rewind();
   XrdDirp->UnLock();
}

/******************************************************************************/
/*                                 R m d i r                                  */
/******************************************************************************/

int XrdPosixXrootd::Rmdir(const char *path)
{
  XrdPosixAdminNew admin(path);

  if (admin.isOK())
     {XrdOucString str(path);
      XrdClientUrlSet url(str);
      if (admin.Admin.Rmdir(url.GetFile().c_str())) return 0;
      return admin.Fault();
     }
  return admin.Result();
}

/******************************************************************************/
/*                                S e e k d i r                               */
/******************************************************************************/

void XrdPosixXrootd::Seekdir(DIR *dirp, long loc)
{
// Sets the current directory position
//
   XrdPosixDir *XrdDirp = findDIR(dirp);
   if (!XrdDirp) return;
   
   if (XrdDirp->getOffset()<0) XrdDirp->nextEntry();  // read the directory
   if (loc >= XrdDirp->getEntries()) loc = XrdDirp->getEntries()-1;
   else if (loc<0) loc = 0;

   XrdDirp->setOffset(loc);
   XrdDirp->UnLock();
}

/******************************************************************************/
/*                                  S t a t                                   */
/******************************************************************************/
  
int XrdPosixXrootd::Stat(const char *path, struct stat *buf)
{
   XrdPosixAdminNew admin(path);
   long st_flags, st_modtime, st_id;
   long long st_size;

// Make sure we connected
//
   if (!admin.isOK()) return admin.Result();

// Extract out path from the url
//
   XrdOucString str(path);
   XrdClientUrlSet url(str);

// Open the file first
//
   if (!admin.Admin.Stat(url.GetFile().c_str(),
                         st_id, st_size, st_flags, st_modtime))
      return admin.Fault();

// Return what little we can
//
   initStat(buf);
   buf->st_size   = st_size;
   buf->st_blocks = buf->st_size/512+1;
   buf->st_atime  = buf->st_mtime = buf->st_ctime = st_modtime;
   buf->st_ino    = st_id;
   buf->st_mode   = mapFlags(st_flags);
   return 0;
}

/******************************************************************************/
/*                                P w r i t e                                 */
/******************************************************************************/
  
ssize_t XrdPosixXrootd::Pwrite(int fildes, const void *buf, size_t nbyte, off_t offset)
{
   XrdPosixFile *fp;
   long long     offs;
   int           iosz;

// Find the file object
//
   if (!(fp = findFP(fildes))) return -1;

// Make sure the size is not too large
//
   if (nbyte > (size_t)0x7fffffff) {Scuttle(fp,EOVERFLOW);}
      else iosz = static_cast<int>(nbyte);

// Issue the write
//
   offs = static_cast<long long>(offset);
   if (!fp->XClient->Write(buf, offs, iosz) && iosz) return Fault(fp);

// All went well
//
   if (offs+iosz > fp->stat.size) fp->stat.size = offs + iosz;
   fp->UnLock();
   return (ssize_t)iosz;
}

/******************************************************************************/
/*                                T e l l d i r                               */
/******************************************************************************/

long XrdPosixXrootd::Telldir(DIR *dirp)
{
// Tells the current directory location
//
   XrdPosixDir *XrdDirp = findDIR(dirp);
   if (!XrdDirp) return -1;

   long pos;
   if (XrdDirp->getOffset()<0) pos = 0;  // dir is open but not read yet
   else pos = XrdDirp->getOffset();
   XrdDirp->UnLock();
   return pos;
}

/******************************************************************************/
/*                                 U n l i n k                                */
/******************************************************************************/

int XrdPosixXrootd::Unlink(const char *path)
{
  XrdPosixAdminNew admin(path);

  if (admin.isOK())
     {XrdOucString str(path);
      XrdClientUrlSet url(str);
      if (admin.Admin.Rm(url.GetFile().c_str())) return 0;
      return admin.Fault();
     }
  return admin.Result();
}

/******************************************************************************/
/*                                 W r i t e                                  */
/******************************************************************************/
  
ssize_t XrdPosixXrootd::Write(int fildes, const void *buf, size_t nbyte)
{
   XrdPosixFile *fp;
   int           iosz;

// Find the file object
//
   if (!(fp = findFP(fildes))) return -1;

// Make sure the size is not too large
//
   if (nbyte > (size_t)0x7fffffff) {Scuttle(fp,EOVERFLOW);}
      else iosz = static_cast<int>(nbyte);

// Issue the write
//
   if (!fp->XClient->Write(buf, fp->Offset(), iosz) && iosz) return Fault(fp);

// All went well
//
   fp->addOffset(iosz, 1);
   fp->UnLock();
   return (ssize_t)iosz;
}
 
/******************************************************************************/
/*                                W r i t e v                                 */
/******************************************************************************/
  
ssize_t XrdPosixXrootd::Writev(int fildes, const struct iovec *iov, int iovcnt)
{
   ssize_t totbytes = 0;
   int i;

// Return the results of the write for each iov segment
//
   for (i = 0; i < iovcnt; i++)
       {if (!Write(fildes,(void *)iov[i].iov_base,(size_t)iov[i].iov_len))
           return -1;
        totbytes += iov[i].iov_len;
       }

// All done
//
   return totbytes;
}
  
/******************************************************************************/
/*                             i s X r o o t d D i r                          */
/******************************************************************************/

bool XrdPosixXrootd::isXrootdDir(DIR *dirp)
{
   if (!dirp) return false;

   for (int i = 0; i <= highDir; i++) 
     if (((XrdPosixDir*)dirp)==myDirs[i]) return true;

   return false;
}
 
/******************************************************************************/
/*                              m a p E r r o r                               */
/******************************************************************************/
  
int XrdPosixXrootd::mapError(int rc)
{
    switch(rc)
       {case kXR_NotFound:      return ENOENT;
        case kXR_NotAuthorized: return EACCES;
        case kXR_IOError:       return EIO;
        case kXR_NoMemory:      return ENOMEM;
        case kXR_NoSpace:       return ENOSPC;
        case kXR_ArgTooLong:    return ENAMETOOLONG;
        case kXR_noserver:      return EHOSTUNREACH;
        case kXR_NotFile:       return ENOTBLK;
        case kXR_isDirectory:   return EISDIR;
        case kXR_FSError:       return ENOSYS;
        default:                return ECANCELED;
       }
}

/******************************************************************************/
/*                              s e t D e b u g                               */
/******************************************************************************/

void XrdPosixXrootd::setDebug(int val)
{
     Debug = static_cast<long>(val);
     setEnv("DebugLevel", val);
}
  
/******************************************************************************/
/*                                s e t E n v                                 */
/******************************************************************************/
  
void XrdPosixXrootd::setEnv(const char *var, const char *val)
{
     EnvPutString(var, val);
}

void XrdPosixXrootd::setEnv(const char *var, long val)
{
     EnvPutInt(var, val);
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                 F a u l t                                  */
/******************************************************************************/

int XrdPosixXrootd::Fault(XrdPosixFile *fp, int complete)
{
   char *etext = fp->XClient->LastServerError()->errmsg;
   int   rc = mapError(fp->XClient->LastServerError()->errnum);

   if (rc != ENOENT && *etext && XrdPosixXrootd::Debug > -2)
      cerr <<"XrdPosix: " <<etext <<endl;

   if (!complete) return rc;
   fp->UnLock();
   errno = rc;
   return -1;
}
  
/******************************************************************************/
/*                                f i n d F P                                 */
/******************************************************************************/
  
XrdPosixFile *XrdPosixXrootd::findFP(int fildes, int glk)
{
   XrdPosixFile *fp;
   int fd;

// Validate the fildes
//
   fd = fildes & FDMask;
   if (fd >= lastFD || fildes < 0 || (fildes & FDLeft) != FDOffs) 
      {errno = EBADF; return (XrdPosixFile *)0;}

// Obtain the file object, if any
//
   myMutex.Lock();
   if (!(fp = myFiles[fd])) {myMutex.UnLock(); errno = EBADF; return fp;}

// Lock the object and unlock the global lock unless it is to be held
//
   fp->Lock();
   if (!glk) myMutex.UnLock();
   return fp;
}

/******************************************************************************/
/*                                f i n d D I R                               */
/******************************************************************************/

XrdPosixDir *XrdPosixXrootd::findDIR(DIR *dirp, int glk)
{
   if (!dirp) { errno = EBADF; return 0; }

// Check if this is really an open directory
//
   XrdPosixDir *XrdDirp = (XrdPosixDir*)dirp;
   myMutex.Lock();
   if (!(myDirs[XrdDirp->dirNo()]==XrdDirp)) {
      myMutex.UnLock();
      errno = EBADF;
      return 0;
   }

// Lock the object and unlock the global lock unless it is to be held
//
   XrdDirp->Lock();
   if (!glk) myMutex.UnLock();
   return XrdDirp;
}
 
/******************************************************************************/
/*                              i n i t S t a t                               */
/******************************************************************************/

void XrdPosixXrootd::initStat(struct stat *buf)
{
   static int initDone = 0;
   static dev_t st_rdev;
   static dev_t st_dev;
   static uid_t myUID = getuid();
   static gid_t myGID = getgid();

// Initialize the xdev fields. This cannot be done in the constructor because
// we mau not yet have resolved the C-library symbols.
//
   if (!initDone) {initDone = 1; initXdev(st_dev, st_rdev);}
   memset(buf, 0, sizeof(struct stat));

// Preset common fields
//
   buf->st_blksize= 64*1024;
   buf->st_dev    = st_dev;
   buf->st_rdev   = st_rdev;
   buf->st_nlink  = 1;
   buf->st_uid    = myUID;
   buf->st_gid    = myGID;
}
  
/******************************************************************************/
/*                              i n i t X d e v                               */
/******************************************************************************/
  
void XrdPosixXrootd::initXdev(dev_t &st_dev, dev_t &st_rdev)
{
   struct stat buf;

// Get the device id for /tmp used by stat()
//
   if (stat("/tmp", &buf)) {st_dev = 0; st_rdev = 0;}
      else {st_dev = buf.st_dev; st_rdev = buf.st_rdev;}
}

/******************************************************************************/
/*                              m a p F l a g s                               */
/******************************************************************************/
  
int XrdPosixXrootd::mapFlags(int flags)
{
   int newflags = 0;

// Map the xroot flags to unix flags
//
   if (flags & kXR_xset)     newflags |= S_IXUSR;
   if (flags & kXR_readable) newflags |= S_IRUSR;
   if (flags & kXR_writable) newflags |= S_IWUSR;
   if (flags & kXR_other)    newflags |= S_IFBLK;
      else if (flags & kXR_isDir) newflags |= S_IFDIR;
              else newflags |= S_IFREG;
   if (flags & kXR_offline) newflags |= S_ISVTX;

   return newflags;
}

/******************************************************************************/
/*                               m a p M o d e                                */
/******************************************************************************/
  
int XrdPosixXrootd::mapMode(mode_t mode)
{  int XMode = 0;

// Map the mode
//
   if (mode & S_IRUSR) XMode |= kXR_ur;
   if (mode & S_IWUSR) XMode |= kXR_uw;
   if (mode & S_IXUSR) XMode |= kXR_ux;
   if (mode & S_IRGRP) XMode |= kXR_gr;
   if (mode & S_IWGRP) XMode |= kXR_gw;
   if (mode & S_IXGRP) XMode |= kXR_gx;
   if (mode & S_IROTH) XMode |= kXR_or;
   if (mode & S_IXOTH) XMode |= kXR_ox;
   return XMode;
}
