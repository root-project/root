/******************************************************************************/
/*                                                                            */
/*                             X r d P s s . c c                              */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

const char *XrdPssCVSID = "$Id$";

/******************************************************************************/
/*                             I n c l u d e s                                */
/******************************************************************************/
  
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <strings.h>
#include <stdio.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef __solaris__
#include <sys/vnode.h>
#endif

#include "XrdVersion.hh"

#include "XrdPss/XrdPss.hh"
#include "XrdPosix/XrdPosixXrootd.hh"

#include "XrdOss/XrdOssError.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

namespace XrdProxy
{
static XrdPssSys   XrdProxySS;
  
       XrdSysError eDest(0, "proxy_");

static const int   PBsz = 3072;
}

using namespace XrdProxy;

/******************************************************************************/
/*                XrdOssGetSS (a.k.a. XrdOssGetStorageSystem)                 */
/******************************************************************************/
  
// This function is called by the OFS layer to retrieve the Storage System
// object. We return our proxy storage system object if configuration succeeded.
//
extern "C"
{
XrdOss *XrdOssGetStorageSystem(XrdOss       *native_oss,
                               XrdSysLogger *Logger,
                               const char   *config_fn,
                               const char   *parms)
{

// Ignore the parms (we accept none for now) and call the init routine
//
   return (XrdProxySS.Init(Logger, config_fn) ? 0 : (XrdOss *)&XrdProxySS);
}
}
 
/******************************************************************************/
/*                      o o s s _ S y s   M e t h o d s                       */
/******************************************************************************/
/******************************************************************************/
/*                                  i n i t                                   */
/******************************************************************************/
  
/*
  Function: Initialize proxy subsystem

  Input:    None

  Output:   Returns zero upon success otherwise (-errno).
*/
int XrdPssSys::Init(XrdSysLogger *lp, const char *configfn)
{
   int NoGo;
   const char *tmp;

// Do the herald thing
//
   eDest.logger(lp);
   eDest.Say("Copr.  2007, Stanford University, Pss Version " XrdVSTRING);

// Initialize the subsystems
//
   tmp = ((NoGo=Configure(configfn)) ? "failed." : "completed.");
   eDest.Say("------ Proxy storage system initialization ", tmp);

// All done.
//
   return NoGo;
}

/******************************************************************************/
/*                                 C h m o d                                  */
/******************************************************************************/
/*
  Function: Change file mode.

  Input:    path        - Is the fully qualified name of the target file.
            mode        - The new mode that the file is to have.

  Output:   Returns XrdOssOK upon success and -errno upon failure.

  Notes:    This function is currently unsupported.
*/

int XrdPssSys::Chmod(const char *path, mode_t mode)
{
// We currently do not support chmod()
//
   return -ENOTSUP;
}

/******************************************************************************/
/*                                c r e a t e                                 */
/******************************************************************************/

/*
  Function: Create a file named `path' with 'file_mode' access mode bits set.

  Input:    path        - The fully qualified name of the file to create.
            access_mode - The Posix access mode bits to be assigned to the file.
                          These bits correspond to the standard Unix permission
                          bits (e.g., 744 == "rwxr--r--").
            env         - Environmental information.
            opts        - Set as follows:
                          XRDOSS_mkpath - create dir path if it does not exist.
                          XRDOSS_new    - the file must not already exist.
                          x00000000     - x are standard open flags (<<8)

  Output:   Returns XrdOssOK upon success; (-errno) otherwise.

  Notes:    We always return ENOTSUP as we really want the create options to be
            promoted to the subsequent open().
*/
int XrdPssSys::Create(const char *tident, const char *path, mode_t Mode,
                        XrdOucEnv &env, int Opts)
{

   return -ENOTSUP;
}

/******************************************************************************/
/*                                 M k d i r                                  */
/******************************************************************************/
/*
  Function: Create a directory

  Input:    path        - Is the fully qualified name of the new directory.
            mode        - The new mode that the directory is to have.
            mkpath      - If true, makes the full path.

  Output:   Returns XrdOssOK upon success and -errno upon failure.

  Notes:    Directories are only created in the local disk cache.
*/

int XrdPssSys::Mkdir(const char *path, mode_t mode, int mkpath)
{
   char pbuff[PBsz];

// Convert path to URL
//
   if (!P2URL(pbuff, PBsz, path)) return -ENAMETOOLONG;

// Simply return the proxied result here (note we do not properly handle mkparh)
//
   return (XrdPosixXrootd::Mkdir(pbuff, mode) ? -errno : XrdOssOK);
}
  
/******************************************************************************/
/*                                R e m d i r                                 */
/******************************************************************************/

/*
  Function: Removes the directory 'path'

  Input:    path      - Is the fully qualified name of the directory to remove.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/
int XrdPssSys::Remdir(const char *path, int Opts)
{

// We currently do not support remdir()
//
   return -ENOTSUP;
}

/******************************************************************************/
/*                                R e n a m e                                 */
/******************************************************************************/

/*
  Function: Renames a file with name 'old_name' to 'new_name'.

  Input:    old_name  - Is the fully qualified name of the file to be renamed.
            new_name  - Is the fully qualified name that the file is to have.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/
int XrdPssSys::Rename(const char *oldname, const char *newname)
{

// We currently do not support rename()
//
   return -ENOTSUP;
}

/******************************************************************************/
/*                                 s t a t                                    */
/******************************************************************************/

/*
  Function: Determine if file 'path' actually exists.

  Input:    path        - Is the fully qualified name of the file to be tested.
            buff        - pointer to a 'stat' structure to hold the attributes
                          of the file.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/

int XrdPssSys::Stat(const char *path, struct stat *buff, int resonly)
{
   char pbuff[PBsz];

// Convert path to URL
//
   if (!P2URL(pbuff, PBsz, path)) return -ENAMETOOLONG;

// Return proxied stat (note we do not properly handle the resonly flag!)
//
   return (XrdPosixXrootd::Stat(pbuff, buff) ? -errno : XrdOssOK);
}

/******************************************************************************/
/*                              T r u n c a t e                               */
/******************************************************************************/
/*
  Function: Truncate a file.

  Input:    path        - Is the fully qualified name of the target file.
            flen        - The new size that the file is to have.

  Output:   Returns XrdOssOK upon success and -errno upon failure.

  Notes:    This function is currently unsupported.
*/

int XrdPssSys::Truncate(const char *path, unsigned long long flen)
{
   char pbuff[PBsz];

// Convert path to URL
//
   if (!P2URL(pbuff, PBsz, path)) return -ENAMETOOLONG;

// Return proxied truncate
//
   return (XrdPosixXrootd::Truncate(pbuff, flen) ? -errno : XrdOssOK);
}
  
/******************************************************************************/
/*                                U n l i n k                                 */
/******************************************************************************/

/*
  Function: Delete a file from the namespace and release it's data storage.

  Input:    path      - Is the fully qualified name of the file to be removed.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/
int XrdPssSys::Unlink(const char *path, int Opts)
{
   char pbuff[PBsz];

// Convert path to URL
//
   if (!P2URL(pbuff, PBsz, path)) return -ENAMETOOLONG;

// Return proxied unlink
//
   return (XrdPosixXrootd::Unlink(pbuff) ? -errno : XrdOssOK);
}

/******************************************************************************/
/*                        P s s D i r   M e t h o d s                         */
/******************************************************************************/
/******************************************************************************/
/*                               o p e n d i r                                */
/******************************************************************************/
  
/*
  Function: Open the directory `path' and prepare for reading.

  Input:    path      - The fully qualified name of the directory to open.

  Output:   Returns XrdOssOK upon success; (-errno) otherwise.
*/
int XrdPssDir::Opendir(const char *dir_path) 
{
   char pbuff[PBsz];

// Convert path to URL
//
   if (!XrdPssSys::P2URL(pbuff, PBsz, dir_path)) return -ENAMETOOLONG;

// Return an error if this object is already open
//
   if (lclfd) return -XRDOSS_E8001;

// Return proxied result
//
   if (!(lclfd = XrdPosixXrootd::Opendir(pbuff))) return -errno;
   return XrdOssOK;
}

/******************************************************************************/
/*                               r e a d d i r                                */
/******************************************************************************/

/*
  Function: Read the next entry if directory associated with this object.

  Input:    buff       - Is the address of the buffer that is to hold the next
                         directory name.
            blen       - Size of the buffer.

  Output:   Upon success, places the contents of the next directory entry
            in buff. When the end of the directory is encountered buff
            will be set to the null string.

            Upon failure, returns a (-errno).

  Warning: The caller must provide proper serialization.
*/
int XrdPssDir::Readdir(char *buff, int blen)
{
   struct dirent *rp;

// Check if this object is actually open
//
   if (!lclfd) return -XRDOSS_E8002;

// Perform proxied result
//
   errno = 0;
   if ((rp = XrdPosixXrootd::Readdir(lclfd)))
      {strlcpy(buff, rp->d_name, blen);
       return XrdOssOK;
      }
   *buff = '\0'; ateof = 1;
   return -errno;
}

/******************************************************************************/
/*                                 C l o s e                                  */
/******************************************************************************/
  
/*
  Function: Close the directory associated with this object.

  Input:    None.

  Output:   Returns XrdOssOK upon success and (errno) upon failure.
*/
int XrdPssDir::Close(long long *retsz)
{

// Make sure this object is open
//
   if (!lclfd) return -XRDOSS_E8002;

// Close whichever handle is open
//
   if (XrdPosixXrootd::Closedir(lclfd)) return -errno;
   lclfd = 0;
   return XrdOssOK;
}

/******************************************************************************/
/*                     o o s s _ F i l e   M e t h o d s                      */
/******************************************************************************/
/******************************************************************************/
/*                                  o p e n                                   */
/******************************************************************************/

/*
  Function: Open the file `path' in the mode indicated by `Mode'.

  Input:    path      - The fully qualified name of the file to open.
            Oflag     - Standard open flags.
            Mode      - Create mode (i.e., rwx).
            env       - Environmental information.

  Output:   XrdOssOK upon success; -errno otherwise.
*/
int XrdPssFile::Open(const char *path, int Oflag, mode_t Mode, XrdOucEnv &Env)
{
   char pbuff[PBsz];

// Convert path to URL
//
   if (!XrdPssSys::P2URL(pbuff, PBsz, path, &Env)) return -ENAMETOOLONG;

// Return an error if this object is already open unless the preceeding call
// was to create the file. For now we ignore the special create flags.
//
   if (fd >= 0)
      {if (fd != 17 || crPath != path) return -XRDOSS_E8003;
          else {fd = 0; crPath = 0; Oflag = crOpts >> 8 | (Oflag & ~O_TRUNC);}
      }

// Return the result of this open
//
   return (fd = XrdPosixXrootd::Open(pbuff,Oflag,Mode)) < 0 ? -errno : XrdOssOK;
}

/******************************************************************************/
/*                                 c l o s e                                  */
/******************************************************************************/

/*
  Function: Close the file associated with this object.

  Input:    None.

  Output:   Returns XrdOssOK upon success aud -errno upon failure.
*/
int XrdPssFile::Close(long long *retsz)
{
    if (fd < 0) return -XRDOSS_E8004;
    if (retsz) *retsz = 0;
    return XrdPosixXrootd::Close(fd) ? -errno : XrdOssOK;
}

/******************************************************************************/
/*                                  r e a d                                   */
/******************************************************************************/

/*
  Function: Preread `blen' bytes from the associated file.

  Input:    offset    - The absolute 64-bit byte offset at which to read.
            blen      - The size to preread.

  Output:   Returns zero read upon success and -errno upon failure.
*/

ssize_t XrdPssFile::Read(off_t offset, size_t blen)
{
     if (fd < 0) return (ssize_t)-XRDOSS_E8004;

     return 0;  // We haven't implemented this yet!
}


/******************************************************************************/
/*                                  r e a d                                   */
/******************************************************************************/

/*
  Function: Read `blen' bytes from the associated file, placing in 'buff'
            the data and returning the actual number of bytes read.

  Input:    buff      - Address of the buffer in which to place the data.
            offset    - The absolute 64-bit byte offset at which to read.
            blen      - The size of the buffer. This is the maximum number
                        of bytes that will be read.

  Output:   Returns the number bytes read upon success and -errno upon failure.
*/

ssize_t XrdPssFile::Read(void *buff, off_t offset, size_t blen)
{
     ssize_t retval;

     if (fd < 0) return (ssize_t)-XRDOSS_E8004;

     return (retval = XrdPosixXrootd::Pread(fd, buff, blen, offset)) < 0
            ? (ssize_t)-errno : retval;
}

/******************************************************************************/
/*                               R e a d R a w                                */
/******************************************************************************/

/*
  Function: Read `blen' bytes from the associated file, placing in 'buff'
            the data and returning the actual number of bytes read.

  Input:    buff      - Address of the buffer in which to place the data.
            offset    - The absolute 64-bit byte offset at which to read.
            blen      - The size of the buffer. This is the maximum number
                        of bytes that will be read.

  Output:   Returns the number bytes read upon success and -errno upon failure.
*/

ssize_t XrdPssFile::ReadRaw(void *buff, off_t offset, size_t blen)
{
     return Read(buff, offset, blen);
}

/******************************************************************************/
/*                                 w r i t e                                  */
/******************************************************************************/

/*
  Function: Write `blen' bytes to the associated file, from 'buff'
            and return the actual number of bytes written.

  Input:    buff      - Address of the buffer from which to get the data.
            offset    - The absolute 64-bit byte offset at which to write.
            blen      - The number of bytes to write from the buffer.

  Output:   Returns the number of bytes written upon success and -errno o/w.
*/

ssize_t XrdPssFile::Write(const void *buff, off_t offset, size_t blen)
{
     ssize_t retval;

     if (fd < 0) return (ssize_t)-XRDOSS_E8004;

     return (retval = XrdPosixXrootd::Pwrite(fd, buff, blen, offset)) < 0
            ? (ssize_t)-errno : retval;
}

/******************************************************************************/
/*                                 f s t a t                                  */
/******************************************************************************/

/*
  Function: Return file status for the associated file.

  Input:    buff      - Pointer to buffer to hold file status.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/

int XrdPssFile::Fstat(struct stat *buff)
{
    if (fd < 0) return -XRDOSS_E8004;

    return (XrdPosixXrootd::Fstat(fd, buff) ? -errno : XrdOssOK);
}

/******************************************************************************/
/*                               f s y n c                                    */
/******************************************************************************/

/*
  Function: Synchronize associated file.

  Input:    None.

  Output:   Returns XrdOssOK upon success and -errno upon failure.
*/
int XrdPssFile::Fsync(void)
{
    if (fd < 0) return -XRDOSS_E8004;

    return (XrdPosixXrootd::Fsync(fd) ? -errno : XrdOssOK);
}

/******************************************************************************/
/*                             f t r u n c a t e                              */
/******************************************************************************/

/*
  Function: Set the length of associated file to 'flen'.

  Input:    flen      - The new size of the file.

  Output:   Returns XrdOssOK upon success and -errno upon failure.

  Notes:    If 'flen' is smaller than the current size of the file, the file
            is made smaller and the data past 'flen' is discarded. If 'flen'
            is larger than the current size of the file, a hole is created
            (i.e., the file is logically extended by filling the extra bytes 
            with zeroes).

            If compiled w/o large file support, only lower 32 bits are used.
            used.

            Currently not supported for proxies.
*/
int XrdPssFile::Ftruncate(unsigned long long flen)
{
    if (fd < 0) return -XRDOSS_E8004;

    return (XrdPosixXrootd::Ftruncate(fd, flen) ?  -errno : XrdOssOK);
}

/******************************************************************************/
/*                               g e t M m a p                                */
/******************************************************************************/
  
/*
  Function: Indicate whether or not file is memory mapped.

  Input:    addr      - Points to an address which will receive the location
                        memory where the file is mapped. If the address is
                        null, true is returned if a mapping exist.

  Output:   Returns the size of the file if it is memory mapped (see above).
            Otherwise, zero is returned and addr is set to zero.
*/
off_t XrdPssFile::getMmap(void **addr)   // Not Supported for proxies
{
   if (addr) *addr = 0;
   return 0;
}
  
/******************************************************************************/
/*                          i s C o m p r e s s e d                           */
/******************************************************************************/
  
/*
  Function: Indicate whether or not file is compressed.

  Input:    cxidp     - Points to a four byte buffer to hold the compression
                        algorithm used if the file is compressed or null.

  Output:   Returns the region size which is 0 if the file is not compressed.
            If cxidp is not null, the algorithm is returned only if the file
            is compressed.
*/
int XrdPssFile::isCompressed(char *cxidp)  // Not supported for proxies
{
    return 0;
}

/******************************************************************************/
/*                     P r i v a t e    S e c t i o n                         */
/******************************************************************************/
/******************************************************************************/
/*                                 P 2 U R L                                  */
/******************************************************************************/
  
int XrdPssSys::P2URL(char *pbuff,int pblen,const char *path,XrdOucEnv *env)
{
     int   theLen, envLen, pathln = strlen(path);
     char *theEnv = 0;

// Calculate the lengths here (include strlen("xrootd://<host>:port/"))
//
   if (env) theEnv = env->Env(envLen);
      else envLen = 0;

   if ((theLen = hdrLen+pathln+(envLen ? envLen+1 : 0)) >= pblen) return 0;

// Copy the data to form complete URL
//
   strcpy(pbuff,        hdrData);
   strcpy(pbuff+hdrLen, path);
   if (envLen)
      {pbuff += (hdrLen + pathln);
       *pbuff++ = '?';
       strcpy(pbuff, theEnv);
      }

   return theLen;
}
