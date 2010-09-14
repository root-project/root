/******************************************************************************/
/*                                                                            */
/*                       X r d X f s N a t i v e . c c                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdSfsNativeCVSID = "$Id$";

#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include <sys/stat.h>

#include "XrdVersion.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSfs/XrdSfsAio.hh"
#include "XrdSfs/XrdSfsNative.hh"

#ifdef AIX
#include <sys/mode.h>
#endif

/******************************************************************************/
/*       O S   D i r e c t o r y   H a n d l i n g   I n t e r f a c e        */
/******************************************************************************/

#ifndef S_IAMB
#define S_IAMB  0x1FF
#endif

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/
  
XrdSysError    *XrdSfsNative::eDest;

/******************************************************************************/
/*            U n i x   F i l e   S y s t e m   I n t e r f a c e             */
/******************************************************************************/

class XrdSfsUFS
{
public:

static int Chmod(const char *fn, mode_t mode) {return chmod(fn, mode);}

static int Close(int fd) {return close(fd);}

static int Mkdir(const char *fn, mode_t mode) {return mkdir(fn, mode);}

static int Open(const char *path, int oflag, mode_t omode)
               {return open(path, oflag, omode);}

static int Rem(const char *fn) {return unlink(fn);}

static int Remdir(const char *fn) {return rmdir(fn);}

static int Rename(const char *ofn, const char *nfn) {return rename(ofn, nfn);}

static int Statfd(int fd, struct stat *buf) {return  fstat(fd, buf);}

static int Statfn(const char *fn, struct stat *buf) {return stat(fn, buf);}

static int Truncate(const char *fn, off_t flen) {return truncate(fn, flen);}
};
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdSfsNative::XrdSfsNative(XrdSysError *ep)
{
  eDest = ep;
}
  
/******************************************************************************/
/*                         G e t F i l e S y s t e m                          */
/******************************************************************************/
  
XrdSfsFileSystem *XrdSfsGetFileSystem(XrdSfsFileSystem *native_fs, 
                                      XrdSysLogger     *lp)
{
 static XrdSysError  Eroute(lp, "XrdSfs");
 static XrdSfsNative myFS(&Eroute);

 Eroute.Say("Copr.  2007 Stanford University/SLAC "
               "sfs (Standard File System) v 9.0n");

 return &myFS;
}

/******************************************************************************/
/*           D i r e c t o r y   O b j e c t   I n t e r f a c e s            */
/******************************************************************************/
/******************************************************************************/
/*                                  o p e n                                   */
/******************************************************************************/
  
int XrdSfsNativeDirectory::open(const char              *dir_path, // In
                                const XrdSecClientName  *client,   // In
                                const char              *info)     // In
/*
  Function: Open the directory `path' and prepare for reading.

  Input:    path      - The fully qualified name of the directory to open.
            cred      - Authentication credentials, if any.
            info      - Opaque information, if any.

  Output:   Returns SFS_OK upon success, otherwise SFS_ERROR.
*/
{
   static const char *epname = "opendir";

// Verify that this object is not already associated with an open directory
//
     if (dh) return
        XrdSfsNative::Emsg(epname, error, EADDRINUSE, 
                             "open directory", dir_path);

// Set up values for this directory object
//
   ateof = 0;
   fname = strdup(dir_path);

// Open the directory and get it's id
//
     if (!(dh = opendir(dir_path))) return
        XrdSfsNative::Emsg(epname,error,errno,"open directory",dir_path);

// All done
//
   return SFS_OK;
}

/******************************************************************************/
/*                             n e x t E n t r y                              */
/******************************************************************************/

const char *XrdSfsNativeDirectory::nextEntry()
/*
  Function: Read the next directory entry.

  Input:    None.

  Output:   Upon success, returns the contents of the next directory entry as
            a null terminated string. Returns a null pointer upon EOF or an
            error. To differentiate the two cases, getErrorInfo will return
            0 upon EOF and an actual error code (i.e., not 0) on error.
*/
{
    static const char *epname = "nextEntry";
    struct dirent *rp;
    int retc;

// Lock the direcrtory and do any required tracing
//
   if (!dh) 
      {XrdSfsNative::Emsg(epname,error,EBADF,"read directory",fname);
       return (const char *)0;
      }

// Check if we are at EOF (once there we stay there)
//
   if (ateof) return (const char *)0;

// Read the next directory entry
//
   errno = 0;
   if ((retc = readdir_r(dh, d_pnt, &rp)))
      {if (retc && errno != 0)
          XrdSfsNative::Emsg(epname,error,retc,"read directory",fname);
       d_pnt->d_name[0] = '\0';
       return (const char *)0;
      }

// Check if we have reached end of file
//
   if (retc || !rp || !d_pnt->d_name[0])
      {ateof = 1;
       error.clear();
       return (const char *)0;
      }

// Return the actual entry
//
   return (const char *)(d_pnt->d_name);
}

/******************************************************************************/
/*                                 c l o s e                                  */
/******************************************************************************/
  
int XrdSfsNativeDirectory::close()
/*
  Function: Close the directory object.

  Input:    cred       - Authentication credentials, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "closedir";

// Release the handle
//
    if (dh && closedir(dh))
       {XrdSfsNative::Emsg(epname, error, errno, "close directory", fname);
        return SFS_ERROR;
       }

// Do some clean-up
//
   if (fname) free(fname);
   dh = (DIR *)0; 
   return SFS_OK;
}

/******************************************************************************/
/*                F i l e   O b j e c t   I n t e r f a c e s                 */
/******************************************************************************/
/******************************************************************************/
/*                                  o p e n                                   */
/******************************************************************************/

int XrdSfsNativeFile::open(const char          *path,      // In
                           XrdSfsFileOpenMode   open_mode, // In
                           mode_t               Mode,      // In
                     const XrdSecClientName    *client,    // In
                     const char                *info)      // In
/*
  Function: Open the file `path' in the mode indicated by `open_mode'.  

  Input:    path      - The fully qualified name of the file to open.
            open_mode - One of the following flag values:
                        SFS_O_RDONLY - Open file for reading.
                        SFS_O_WRONLY - Open file for writing.
                        SFS_O_RDWR   - Open file for update
                        SFS_O_CREAT  - Create the file open in RDWR mode
                        SFS_O_TRUNC  - Trunc  the file open in RDWR mode
            Mode      - The Posix access mode bits to be assigned to the file.
                        These bits correspond to the standard Unix permission
                        bits (e.g., 744 == "rwxr--r--"). Mode may also conatin
                        SFS_O_MKPTH is the full path is to be created. The
                        agument is ignored unless open_mode = SFS_O_CREAT.
            client    - Authentication credentials, if any.
            info      - Opaque information to be used as seen fit.

  Output:   Returns OOSS_OK upon success, otherwise SFS_ERROR is returned.
*/
{
   static const char *epname = "open";
   const int AMode = S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH; // 775
   char *opname;
   mode_t acc_mode = Mode & S_IAMB;
   int retc, open_flag = 0;
   struct stat buf;

// Verify that this object is not already associated with an open file
//
   if (oh >= 0)
      return XrdSfsNative::Emsg(epname,error,EADDRINUSE,"open file",path);
   fname = strdup(path);

// Set the actual open mode
//
   switch(open_mode & (SFS_O_RDONLY | SFS_O_WRONLY | SFS_O_RDWR))
   {
   case SFS_O_RDONLY: open_flag = O_RDONLY; break;
   case SFS_O_WRONLY: open_flag = O_WRONLY; break;
   case SFS_O_RDWR:   open_flag = O_RDWR;   break;
   default:           open_flag = O_RDONLY; break;
   }

// Prepare to create or open the file, as needed
//
   if (open_mode & SFS_O_CREAT)
      {open_flag  = O_RDWR | O_CREAT | O_EXCL;
       opname = (char *)"create";
       if ((Mode & SFS_O_MKPTH) && (retc = XrdSfsNative::Mkpath(path,AMode,info)))
          return XrdSfsNative::Emsg(epname,error,retc,"create path for",path);
      } else if (open_mode & SFS_O_TRUNC)
                {open_flag  = O_RDWR | O_CREAT | O_TRUNC;
                 opname = (char *)"truncate";
                } else opname = (char *)"open";

// Open the file and make sure it is a file
//
   if ((oh = XrdSfsUFS::Open(path, open_flag, acc_mode)) >= 0)
      {do {retc = XrdSfsUFS::Statfd(oh, &buf);} while(retc && errno == EINTR);
       if (!retc && !(buf.st_mode & S_IFREG))
          {close(); oh = (buf.st_mode & S_IFDIR ? -EISDIR : -ENOTBLK);}
      } else {
       oh = -errno;
       if (errno == EEXIST)
          {do {retc = XrdSfsUFS::Statfn(path, &buf);}
              while(retc && errno == EINTR);
           if (!retc && (buf.st_mode & S_IFDIR)) oh = -EISDIR;
          }
      }

// All done.
//
   if (oh < 0) return XrdSfsNative::Emsg(epname, error, oh, opname, path);
   return SFS_OK;
}

/******************************************************************************/
/*                                 c l o s e                                  */
/******************************************************************************/

int XrdSfsNativeFile::close()
/*
  Function: Close the file object.

  Input:    None

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "close";

// Release the handle and return
//
    if (oh >= 0  && XrdSfsUFS::Close(oh))
       return XrdSfsNative::Emsg(epname, error, errno, "close", fname);
    oh = -1;
    if (fname) {free(fname); fname = 0;}
    return SFS_OK;
}

/******************************************************************************/
/*                                  f c t l                                   */
/******************************************************************************/

int      XrdSfsNativeFile::fctl(const int               cmd,
                                const char             *args,
                                      XrdOucErrInfo    &out_error)
{
// See if we can do this
//
   if (cmd == SFS_FCTL_GETFD)
      {out_error.setErrCode(oh);
       return SFS_OK;
      }

// We don't support this
//
   out_error.setErrInfo(EEXIST, "fctl operation not supported");
   return SFS_ERROR;
}
  
/******************************************************************************/
/*                                  r e a d                                   */
/******************************************************************************/

XrdSfsXferSize XrdSfsNativeFile::read(XrdSfsFileOffset  offset,    // In
                                      char             *buff,      // Out
                                      XrdSfsXferSize    blen)      // In
/*
  Function: Read `blen' bytes at `offset' into 'buff' and return the actual
            number of bytes read.

  Input:    offset    - The absolute byte offset at which to start the read.
            buff      - Address of the buffer in which to place the data.
            blen      - The size of the buffer. This is the maximum number
                        of bytes that will be read from 'fd'.

  Output:   Returns the number of bytes read upon success and SFS_ERROR o/w.
*/
{
   static const char *epname = "read";
   XrdSfsXferSize nbytes;

// Make sure the offset is not too large
//
#if _FILE_OFFSET_BITS!=64
   if (offset >  0x000000007fffffff)
      return XrdSfsNative::Emsg(epname, error, EFBIG, "read", fname);
#endif

// Read the actual number of bytes
//
   do { nbytes = pread(oh, (void *)buff, (size_t)blen, (off_t)offset); }
        while(nbytes < 0 && errno == EINTR);

   if (nbytes  < 0)
      return XrdSfsNative::Emsg(epname, error, errno, "read", fname);

// Return number of bytes read
//
   return nbytes;
}
  
/******************************************************************************/
/*                              r e a d   A I O                               */
/******************************************************************************/
  
int XrdSfsNativeFile::read(XrdSfsAio *aiop)
{

// Execute this request in a synchronous fashion
//
   aiop->Result = this->read((XrdSfsFileOffset)aiop->sfsAio.aio_offset,
                                       (char *)aiop->sfsAio.aio_buf,
                               (XrdSfsXferSize)aiop->sfsAio.aio_nbytes);
   aiop->doneRead();
   return 0;
}

/******************************************************************************/
/*                                 w r i t e                                  */
/******************************************************************************/

XrdSfsXferSize XrdSfsNativeFile::write(XrdSfsFileOffset   offset,    // In
                                       const char        *buff,      // In
                                       XrdSfsXferSize     blen)      // In
/*
  Function: Write `blen' bytes at `offset' from 'buff' and return the actual
            number of bytes written.

  Input:    offset    - The absolute byte offset at which to start the write.
            buff      - Address of the buffer from which to get the data.
            blen      - The size of the buffer. This is the maximum number
                        of bytes that will be written to 'fd'.

  Output:   Returns the number of bytes written upon success and SFS_ERROR o/w.

  Notes:    An error return may be delayed until the next write(), close(), or
            sync() call.
*/
{
   static const char *epname = "write";
   XrdSfsXferSize nbytes;

// Make sure the offset is not too large
//
#if _FILE_OFFSET_BITS!=64
   if (offset >  0x000000007fffffff)
      return XrdSfsNative::Emsg(epname, error, EFBIG, "write", fname);
#endif

// Write the requested bytes
//
   do { nbytes = pwrite(oh, (void *)buff, (size_t)blen, (off_t)offset); }
        while(nbytes < 0 && errno == EINTR);

   if (nbytes  < 0)
      return XrdSfsNative::Emsg(epname, error, errno, "write", fname);

// Return number of bytes written
//
   return nbytes;
}

/******************************************************************************/
/*                             w r i t e   A I O                              */
/******************************************************************************/
  
int XrdSfsNativeFile::write(XrdSfsAio *aiop)
{

// Execute this request in a synchronous fashion
//
   aiop->Result = this->write((XrdSfsFileOffset)aiop->sfsAio.aio_offset,
                                        (char *)aiop->sfsAio.aio_buf,
                                (XrdSfsXferSize)aiop->sfsAio.aio_nbytes);
   aiop->doneWrite();
   return 0;
}
  
/******************************************************************************/
/*                                  s t a t                                   */
/******************************************************************************/

int XrdSfsNativeFile::stat(struct stat     *buf)         // Out
/*
  Function: Return file status information

  Input:    buf         - The stat structiure to hold the results

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "stat";

// Execute the function
//
   if (XrdSfsUFS::Statfd(oh, buf))
      return XrdSfsNative::Emsg(epname, error, errno, "state", fname);

// All went well
//
   return SFS_OK;
}

/******************************************************************************/
/*                                  s y n c                                   */
/******************************************************************************/

int XrdSfsNativeFile::sync()
/*
  Function: Commit all unwritten bytes to physical media.

  Input:    None

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "sync";

// Perform the function
//
   if (fsync(oh))
      return XrdSfsNative::Emsg(epname,error,errno,"synchronize",fname);

// All done
//
   return SFS_OK;
}

/******************************************************************************/
/*                              s y n c   A I O                               */
/******************************************************************************/
  
int XrdSfsNativeFile::sync(XrdSfsAio *aiop)
{

// Execute this request in a synchronous fashion
//
   aiop->Result = this->sync();
   aiop->doneWrite();
   return 0;
}

/******************************************************************************/
/*                              t r u n c a t e                               */
/******************************************************************************/

int XrdSfsNativeFile::truncate(XrdSfsFileOffset  flen)  // In
/*
  Function: Set the length of the file object to 'flen' bytes.

  Input:    flen      - The new size of the file.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.

  Notes:    If 'flen' is smaller than the current size of the file, the file
            is made smaller and the data past 'flen' is discarded. If 'flen'
            is larger than the current size of the file, a hole is created
            (i.e., the file is logically extended by filling the extra bytes 
            with zeroes).
*/
{
   static const char *epname = "trunc";

// Make sure the offset is not too larg
//
   if (sizeof(off_t) < sizeof(flen) && flen >  0x000000007fffffff)
      return XrdSfsNative::Emsg(epname, error, EFBIG, "truncate", fname);

// Perform the function
//
   if (ftruncate(oh, flen))
      return XrdSfsNative::Emsg(epname, error, errno, "truncate", fname);

// All done
//
   return SFS_OK;
}

/******************************************************************************/
/*         F i l e   S y s t e m   O b j e c t   I n t e r f a c e s          */
/******************************************************************************/
/******************************************************************************/
/*                                 c h m o d                                  */
/******************************************************************************/

int XrdSfsNative::chmod(const char             *path,    // In
                              XrdSfsMode        Mode,    // In
                              XrdOucErrInfo    &error,   // Out
                        const XrdSecClientName *client,  // In
                        const char             *info)    // In
/*
  Function: Change the mode on a file or directory.

  Input:    path      - Is the fully qualified name of the file to be removed.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "chmod";
   mode_t acc_mode = Mode & S_IAMB;

// Perform the actual deletion
//
   if (XrdSfsUFS::Chmod(path, acc_mode) )
      return XrdSfsNative::Emsg(epname,error,errno,"change mode on",path);

// All done
//
    return SFS_OK;
}
  
/******************************************************************************/
/*                                e x i s t s                                 */
/******************************************************************************/

int XrdSfsNative::exists(const char                *path,        // In
                               XrdSfsFileExistence &file_exists, // Out
                               XrdOucErrInfo       &error,       // Out
                         const XrdSecClientName    *client,      // In
                         const char                *info)        // In
/*
  Function: Determine if file 'path' actually exists.

  Input:    path        - Is the fully qualified name of the file to be tested.
            file_exists - Is the address of the variable to hold the status of
                          'path' when success is returned. The values may be:
                          XrdSfsFileExistsIsDirectory - file not found but path is valid.
                          XrdSfsFileExistsIsFile      - file found.
                          XrdSfsFileExistsIsNo        - neither file nor directory.
            einfo       - Error information object holding the details.
            client      - Authentication credentials, if any.
            info        - Opaque information, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.

  Notes:    When failure occurs, 'file_exists' is not modified.
*/
{
   static const char *epname = "exists";
   struct stat fstat;

// Now try to find the file or directory
//
   if (!XrdSfsUFS::Statfn(path, &fstat) )
      {     if (S_ISDIR(fstat.st_mode)) file_exists=XrdSfsFileExistIsDirectory;
       else if (S_ISREG(fstat.st_mode)) file_exists=XrdSfsFileExistIsFile;
       else                             file_exists=XrdSfsFileExistNo;
       return SFS_OK;
      }
   if (errno == ENOENT)
      {file_exists=XrdSfsFileExistNo;
       return SFS_OK;
      }

// An error occured, return the error info
//
   return XrdSfsNative::Emsg(epname, error, errno, "locate", path);
}

/******************************************************************************/
/*                                 f s c t l                                  */
/******************************************************************************/

int XrdSfsNative::fsctl(const int               cmd,
                        const char             *args,
                              XrdOucErrInfo    &out_error,
                        const XrdSecClientName *client)
{
    out_error.setErrInfo(ENOTSUP, "Operation not supported.");
    return SFS_ERROR;
}
  
/******************************************************************************/
/*                            g e t V e r s i o n                             */
/******************************************************************************/

const char *XrdSfsNative::getVersion() {return XrdVERSION;}

/******************************************************************************/
/*                                 m k d i r                                  */
/******************************************************************************/

int XrdSfsNative::mkdir(const char             *path,    // In
                              XrdSfsMode        Mode,    // In
                              XrdOucErrInfo    &error,   // Out
                        const XrdSecClientName *client,  // In
                        const char             *info)    // In
/*
  Function: Create a directory entry.

  Input:    path      - Is the fully qualified name of the file to be removed.
            Mode      - Is the POSIX mode setting for the directory. If the
                        mode contains SFS_O_MKPTH, the full path is created.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "mkdir";
   mode_t acc_mode = Mode & S_IAMB;

// Create the path if it does not already exist
//
   if (Mode & SFS_O_MKPTH) Mkpath(path, acc_mode, info);

// Perform the actual deletion
//
   if (XrdSfsUFS::Mkdir(path, acc_mode) )
      return XrdSfsNative::Emsg(epname,error,errno,"create directory",path);

// All done
//
    return SFS_OK;
}

/******************************************************************************/
/*                                M k p a t h                                 */
/******************************************************************************/
/*
  Function: Create a directory path

  Input:    path        - Is the fully qualified name of the new path.
            mode        - The new mode that each new directory is to have.
            info        - Opaque information, of any.

  Output:   Returns 0 upon success and -errno upon failure.
*/

int XrdSfsNative::Mkpath(const char *path, mode_t mode, const char *info)
{
    char actual_path[MAXPATHLEN], *local_path, *next_path;
    unsigned int plen;
    struct stat buf;

// Extract out the path we should make
//
   if (!(plen = strlen(path))) return -ENOENT;
   if (plen >= sizeof(actual_path)) return -ENAMETOOLONG;
   strcpy(actual_path, path);
   if (actual_path[plen-1] == '/') actual_path[plen-1] = '\0';

// Typically, the path exist. So, do a quick check before launching into it
//
   if (!(local_path = rindex(actual_path, (int)'/'))
   ||  local_path == actual_path) return 0;
   *local_path = '\0';
   if (!XrdSfsUFS::Statfn(actual_path, &buf)) return 0;
   *local_path = '/';

// Start creating directories starting with the root. Notice that we will not
// do anything with the last component. The caller is responsible for that.
//
   local_path = actual_path+1;
   while((next_path = index(local_path, int('/'))))
        {*next_path = '\0';
         if (XrdSfsUFS::Mkdir(actual_path,mode) && errno != EEXIST)
            return -errno;
         *next_path = '/';
         local_path = next_path+1;
        }

// All done
//
   return 0;
}

/******************************************************************************/
/*                                   r e m                                    */
/******************************************************************************/
  
int XrdSfsNative::rem(const char             *path,    // In
                            XrdOucErrInfo    &error,   // Out
                      const XrdSecClientName *client,  // In
                      const char             *info)    // In
/*
  Function: Delete a file from the namespace.

  Input:    path      - Is the fully qualified name of the file to be removed.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "rem";

// Perform the actual deletion
//
    if (XrdSfsUFS::Rem(path) )
       return XrdSfsNative::Emsg(epname, error, errno, "remove", path);

// All done
//
    return SFS_OK;
}

/******************************************************************************/
/*                                r e m d i r                                 */
/******************************************************************************/

int XrdSfsNative::remdir(const char             *path,    // In
                               XrdOucErrInfo    &error,   // Out
                         const XrdSecClientName *client,  // In
                         const char             *info)    // In
/*
  Function: Delete a directory from the namespace.

  Input:    path      - Is the fully qualified name of the dir to be removed.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "remdir";

// Perform the actual deletion
//
    if (XrdSfsUFS::Remdir(path) )
       return XrdSfsNative::Emsg(epname, error, errno, "remove", path);

// All done
//
    return SFS_OK;
}

/******************************************************************************/
/*                                r e n a m e                                 */
/******************************************************************************/

int XrdSfsNative::rename(const char             *old_name,  // In
                         const char             *new_name,  // In
                               XrdOucErrInfo    &error,     //Out
                         const XrdSecClientName *client,    // In
                         const char             *infoO,     // In
                         const char             *infoN)     // In
/*
  Function: Renames a file/directory with name 'old_name' to 'new_name'.

  Input:    old_name  - Is the fully qualified name of the file to be renamed.
            new_name  - Is the fully qualified name that the file is to have.
            error     - Error information structure, if an error occurs.
            client    - Authentication credentials, if any.
            info      - old_name opaque information, if any.
            info      - new_name opaque information, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "rename";

// Perform actual rename operation
//
   if (XrdSfsUFS::Rename(old_name, new_name) )
      return XrdSfsNative::Emsg(epname, error, errno, "rename", old_name);

// All done
//
   return SFS_OK;
}
  
/******************************************************************************/
/*                                  s t a t                                   */
/******************************************************************************/

int XrdSfsNative::stat(const char              *path,        // In
                             struct stat       *buf,         // Out
                             XrdOucErrInfo     &error,       // Out
                       const XrdSecClientName  *client,      // In
                       const char              *info)        // In
/*
  Function: Get info on 'path'.

  Input:    path        - Is the fully qualified name of the file to be tested.
            buf         - The stat structiure to hold the results
            error       - Error information object holding the details.
            client      - Authentication credentials, if any.
            info        - Opaque information, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   static const char *epname = "stat";

// Execute the function
//
   if (XrdSfsUFS::Statfn(path, buf) )
      return XrdSfsNative::Emsg(epname, error, errno, "state", path);

// All went well
//
   return SFS_OK;
}

/******************************************************************************/
/*                              t r u n c a t e                               */
/******************************************************************************/
  
int XrdSfsNative::truncate(const char             *path,    // In
                                 XrdSfsFileOffset  flen,    // In
                                 XrdOucErrInfo    &error,   // Out
                           const XrdSecClientName *client,  // In
                           const char             *info)    // In
/*
  Function: Set the length of the file object to 'flen' bytes.

  Input:    path      - The path to the file.
            flen      - The new size of the file.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.

  Notes:    If 'flen' is smaller than the current size of the file, the file
            is made smaller and the data past 'flen' is discarded. If 'flen'
            is larger than the current size of the file, a hole is created
            (i.e., the file is logically extended by filling the extra bytes 
            with zeroes).
*/
{
   static const char *epname = "trunc";

// Make sure the offset is not too larg
//
   if (sizeof(off_t) < sizeof(flen) && flen >  0x000000007fffffff)
      return XrdSfsNative::Emsg(epname, error, EFBIG, "truncate", path);

// Perform the function
//
   if (XrdSfsUFS::Truncate(path, flen) )
      return XrdSfsNative::Emsg(epname, error, errno, "truncate", path);

// All done
//
   return SFS_OK;
}

/******************************************************************************/
/*                                  E m s g                                   */
/******************************************************************************/

int XrdSfsNative::Emsg(const char    *pfx,    // Message prefix value
                       XrdOucErrInfo &einfo,  // Place to put text & error code
                       int            ecode,  // The error code
                       const char    *op,     // Operation being performed
                       const char    *target) // The target (e.g., fname)
{
    char *etext, buffer[MAXPATHLEN+80], unkbuff[64];

// Get the reason for the error
//
   if (ecode < 0) ecode = -ecode;
   if (!(etext = strerror(ecode)))
      {sprintf(unkbuff, "reason unknown (%d)", ecode); etext = unkbuff;}

// Format the error message
//
    snprintf(buffer,sizeof(buffer),"Unable to %s %s; %s", op, target, etext);

// Print it out if debugging is enabled
//
#ifndef NODEBUG
   eDest->Emsg(pfx, buffer);
#endif

// Place the error message in the error object and return
//
    einfo.setErrInfo(ecode, buffer);

    return SFS_ERROR;
}
