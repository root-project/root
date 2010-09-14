/******************************************************************************/
/*                                                                            */
/*                             X r d B w m . c c                              */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdBwmCVSID = "$Id$";

#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "XrdVersion.hh"

#include "XrdBwm/XrdBwm.hh"
#include "XrdBwm/XrdBwmTrace.hh"

#include "XrdAcc/XrdAccAuthorize.hh"

#include "XrdNet/XrdNetDNS.hh"

#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucTrace.hh"

#include "XrdSec/XrdSecEntity.hh"

#include "XrdSfs/XrdSfsAio.hh"
#include "XrdSfs/XrdSfsInterface.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                  E r r o r   R o u t i n g   O b j e c t                   */
/******************************************************************************/

XrdSysError      BwmEroute(0);

XrdOucTrace      BwmTrace(&BwmEroute);

/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/
  
XrdBwmHandle     *XrdBwm::dummyHandle;

/******************************************************************************/
/*                    F i l e   S y s t e m   O b j e c t                     */
/******************************************************************************/
  
XrdBwm XrdBwmFS;

/******************************************************************************/
/*                    X r d B w m   C o n s t r u c t o r                     */
/******************************************************************************/

XrdBwm::XrdBwm()
{
   unsigned int myIPaddr = 0;
   char buff[256], *bp;
   int myPort, i;

// Establish defaults
//
   Authorization = 0;
   Authorize     = 0;
   AuthLib       = 0;
   AuthParm      = 0;
   Logger        = 0;
   PolLib        = 0;
   PolParm       = 0;
   PolSlotsIn    = 1;
   PolSlotsOut   = 1;

// Obtain port number we will be using
//
   myPort = (bp = getenv("XRDPORT")) ? strtol(bp, (char **)NULL, 10) : 0;

// Establish our hostname and IPV4 address
//
   HostName      = XrdNetDNS::getHostName();
   if (!XrdNetDNS::Host2IP(HostName, &myIPaddr)) myIPaddr = 0x7f000001;
   strcpy(buff, "[::"); bp = buff+3;
   bp += XrdNetDNS::IP2String(myIPaddr, 0, bp, 128);
   *bp++ = ']'; *bp++ = ':';
   sprintf(bp, "%d", myPort);
   locResp = strdup(buff); locRlen = strlen(buff);
   for (i = 0; HostName[i] && HostName[i] != '.'; i++);
   HostName[i] = '\0';
   HostPref = strdup(HostName);
   HostName[i] = '.';
   myDomain = &HostName[i+1];
   myDomLen = strlen(myDomain);

// Set the configuration file name abd dummy handle
//
   ConfigFN = 0;
   dummyHandle = XrdBwmHandle::Alloc("*", "/", "?", "?", 0);
}
  
/******************************************************************************/
/*                X r d B w m F i l e   C o n s t r u c t o r                 */
/******************************************************************************/

XrdBwmFile::XrdBwmFile(const char *user) : XrdSfsFile(user)
{
   oh = XrdBwm::dummyHandle;
   tident = (user ? user : "");
}
  
/******************************************************************************/
/*                         G e t F i l e S y s t e m                          */
/******************************************************************************/
  
extern "C"
{
XrdSfsFileSystem *XrdSfsGetFileSystem(XrdSfsFileSystem *native_fs, 
                                      XrdSysLogger     *lp,
                                      const char       *configfn)
{
// Do the herald thing
//
   BwmEroute.SetPrefix("bwm_");
   BwmEroute.logger(lp);
   BwmEroute.Say("Copr.  2008 Stanford University, Bwm Version " XrdVSTRING);

// Initialize the subsystems
//
   XrdBwmFS.ConfigFN = (configfn && *configfn ? strdup(configfn) : 0);
   if ( XrdBwmFS.Configure(BwmEroute) ) return 0;

// All done, we can return the callout vector to these routines.
//
   return &XrdBwmFS;
}
}

/******************************************************************************/
/*                                                                            */
/*           D i r e c t o r y   O b j e c t   I n t e r f a c e s            */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/
/*                                  o p e n                                   */
/******************************************************************************/

int XrdBwmDirectory::open(const char              *dir_path, // In
                          const XrdSecEntity      *client,   // In
                          const char              *info)      // In
/*
  Function: Open the directory `path' and prepare for reading.

  Input:    path      - The fully qualified name of the directory to open.
            client    - Authentication credentials, if any.
            info      - Opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success, otherwise SFS_ERROR.

  Notes: 1. Currently, function not supported.
*/
{
// Return an error
//
   return XrdBwmFS.Emsg("opendir", error, ENOTDIR, "open directory", dir_path);
}

/******************************************************************************/
/*                             n e x t E n t r y                              */
/******************************************************************************/

const char *XrdBwmDirectory::nextEntry()
/*
  Function: Read the next directory entry.

  Input:    n/a

  Output:   n/a
*/
{
// Return an error
//
   XrdBwmFS.Emsg("readdir", error, EBADF, "read directory");
   return 0;
}

/******************************************************************************/
/*                                 c l o s e                                  */
/******************************************************************************/
  
int XrdBwmDirectory::close()
/*
  Function: Close the directory object.

  Input:    n/a

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
// Return an error
//
   XrdBwmFS.Emsg("closedir", error, EBADF, "close directory");
   return SFS_ERROR;
}

/******************************************************************************/
/*                                                                            */
/*                F i l e   O b j e c t   I n t e r f a c e s                 */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/
/*                                  o p e n                                   */
/******************************************************************************/

int XrdBwmFile::open(const char          *path,      // In
                     XrdSfsFileOpenMode   open_mode, // In
                     mode_t               Mode,      // In
               const XrdSecEntity        *client,    // In
               const char                *info)      // In
/*
  Function: Open the file `path' in the mode indicated by `open_mode'.  

  Input:    path      - The fully qualified name of the file to open.
                        The path must start with "/_bwm_" and the lfn that
                        will eventually be opened start at the next slash.
            open_mode - One of the following flag values:
                        SFS_O_RDONLY - Open file for reading.
                        SFS_O_WRONLY - Open file for writing.           n/a
                        SFS_O_RDWR   - Open file for update             n/a
                        SFS_O_CREAT  - Create the file open in RW mode  n/a
                        SFS_O_TRUNC  - Trunc  the file open in RW mode  n/a
            Mode      - The Posix access mode bits to be assigned to the file.
                        These bits are ignored.
            client    - Authentication credentials, if any.
            info      - Opaque information:
                        bwm.src=<src  host>
                        bwm.dst=<dest host>

  Output:   Returns SFS_OK upon success, otherwise SFS_ERROR is returned.
*/
{
   EPNAME("open");
   XrdBwmHandle *hP;
   int incomming;
   const char *miss, *theUsr, *theSrc, *theDst=0, *theLfn=0, *lclNode, *rmtNode;
   XrdOucEnv Open_Env(info);

// Trace entry
//
   ZTRACE(calls,std::hex <<open_mode <<std::dec <<" fn=" <<path);

// Verify that this object is not already associated with an open file
//
   XrdBwmFS.ocMutex.Lock();
   if (oh != XrdBwm::dummyHandle)
      {XrdBwmFS.ocMutex.UnLock();
       return XrdBwmFS.Emsg("open",error,EADDRINUSE,"open file",path);
      }
   XrdBwmFS.ocMutex.UnLock();

// Verify that the file is being opened in r/w mode only!
//
   if (!(open_mode & SFS_O_RDWR))
      return XrdBwmFS.Emsg("open", error, EINVAL, "open", path);

// Apply security. Note that we reject r/w access but apply r/o access
// restrictions if so wanted.
//
   if (client && XrdBwmFS.Authorization
   &&  !XrdBwmFS.Authorization->Access(client, path, AOP_Update, &Open_Env))
      return XrdBwmFS.Emsg("open", error, EACCES, "open", path);

// Make sure that all of the relevant information is present
//
        if (!(theSrc = Open_Env.Get("bwm.src"))) miss = "bwm.src";
   else if (!(theDst = Open_Env.Get("bwm.dst"))) miss = "bwm.dst";
   else if (!(theLfn = index(path+1,'/'))
        ||  !(*(theLfn+1)))                     miss = "lfn";
   else                                         miss = 0;

   if (miss) return XrdBwmFS.Emsg("open", error, miss, "open", path);
   theUsr = error.getErrUser();

// Determine the direction of flow
//
        if (XrdNetDNS::isDomain(theSrc, XrdBwmFS.myDomain, XrdBwmFS.myDomLen))
           {incomming = 0; lclNode = theSrc; rmtNode = theDst;}
   else if (XrdNetDNS::isDomain(theDst, XrdBwmFS.myDomain, XrdBwmFS.myDomLen))
           {incomming = 1; lclNode = theDst; rmtNode = theSrc;}
   else return XrdBwmFS.Emsg("open", error, EREMOTE, "open", path);

// Get a handle for this file.
//
   if (!(hP = XrdBwmHandle::Alloc(theUsr,theLfn,lclNode,rmtNode,incomming)))
      return XrdBwmFS.Stall(error, 13, path);

// All done
//
   XrdBwmFS.ocMutex.Lock(); oh = hP; XrdBwmFS.ocMutex.UnLock();
   return SFS_OK;
}

/******************************************************************************/
/*                                 c l o s e                                  */
/******************************************************************************/

int XrdBwmFile::close()  // In
/*
  Function: Close the file object.

  Input:    n/a

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   EPNAME("close");
   XrdBwmHandle *hP;

// Trace the call
//
   FTRACE(calls, "close" <<oh->Name());

// Verify the handle (we briefly maintain a global lock)
//
   XrdBwmFS.ocMutex.Lock();
   if (oh == XrdBwm::dummyHandle)
      {XrdBwmFS.ocMutex.UnLock(); return SFS_OK;}
   hP = oh; oh = XrdBwm::dummyHandle;
   XrdBwmFS.ocMutex.UnLock();

// Now retire it and possibly return the token
//
   hP->Retire();

// All done
//
   return SFS_OK;
}

/******************************************************************************/
/*                                  f c t l                                   */
/******************************************************************************/
  
int            XrdBwmFile::fctl(const int               cmd,
                                const char             *args,
                                      XrdOucErrInfo    &out_error)
/*
  Function: perform request control operation.

  Input:    cmd       - The operation:
                        SFS_FCTL_GETFD - not supported.
                        SFS_FCTL_STATV - returns visa information
            args      - Dependent on the cmd.
            out_error - Place where response goes.

  Output:   Returns SFS_OK upon success and SFS_ERROR o/w.
*/
{

// Make sure the file is open
//
   if (oh == XrdBwm::dummyHandle)
      return XrdBwmFS.Emsg("fctl", out_error, EBADF, "fctl file");

// Scan through the fctl operations
//
   switch(cmd)
         {case SFS_FCTL_GETFD:  out_error.setErrInfo(-1,"");
                                return SFS_OK;
          case SFS_FCTL_STATV:  return oh->Activate(out_error);
          default:              break;
         }

// Invalid fctl
//
   out_error.setErrInfo(EINVAL, "invalid fctl command");
   return SFS_ERROR;
}

/******************************************************************************/
/*                                  r e a d                                   */
/******************************************************************************/

int            XrdBwmFile::read(XrdSfsFileOffset  offset,    // In
                                XrdSfsXferSize    blen)      // In
/*
  Function: Preread `blen' bytes at `offset'

  Input:    offset    - The absolute byte offset at which to start the read.
            blen      - The amount to preread.

  Output:   Returns SFS_OK upon success and SFS_ERROR o/w.
*/
{
   EPNAME("read");

// Perform required tracing
//
   FTRACE(calls,"preread " <<blen <<"@" <<offset);

// Return number of bytes read
//
   return 0;
}
  
/******************************************************************************/
/*                                  r e a d                                   */
/******************************************************************************/

XrdSfsXferSize XrdBwmFile::read(XrdSfsFileOffset  offset,    // In
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

  Notes: 1. Currently, we have no information so we always return 0 bytes.
*/
{
   EPNAME("read");

// Perform required tracing
//
   FTRACE(calls,blen <<"@" <<offset);

// Return number of bytes read
//
   return 0;
}
  
/******************************************************************************/
/*                              r e a d   A I O                               */
/******************************************************************************/
  
/*
  Function: Read `blen' bytes at `offset' into 'buff' and return the actual
            number of bytes read using asynchronous I/O, if possible.

  Output:   Returns the 0 if successfullt queued, otherwise returns an error.
            The underlying implementation will convert the request to
            synchronous I/O is async mode is not possible.
*/

int XrdBwmFile::read(XrdSfsAio *aiop)
{

// Async mode not supported.
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

XrdSfsXferSize XrdBwmFile::write(XrdSfsFileOffset  offset,    // In
                                 const char       *buff,      // Out
                                 XrdSfsXferSize    blen)      // In
/*
  Function: Write `blen' bytes at `offset' from 'buff' and return the actual
            number of bytes written.

  Input:    offset    - The absolute byte offset at which to start the write.
            buff      - Address of the buffer from which to get the data.
            blen      - The size of the buffer. This is the maximum number
                        of bytes that will be written to 'fd'.

  Output:   Returns the number of bytes written upon success and SFS_ERROR o/w.

  Notes: 1. An error return may be delayed until the next write(), close(), or
            sync() call.
         2. Currently, we do not accept write activated commands.
*/
{
   EPNAME("write");

// Perform any required tracing
//
   FTRACE(calls, blen <<"@" <<offset);

// Return number of bytes written
//
   return 0;
}

/******************************************************************************/
/*                             w r i t e   A I O                              */
/******************************************************************************/
  
// For now, this reverts to synchronous I/O
//
int XrdBwmFile::write(XrdSfsAio *aiop)
{

// Async mode not supported.
//
   aiop->Result = this->write((XrdSfsFileOffset)aiop->sfsAio.aio_offset,
                                        (char *)aiop->sfsAio.aio_buf,
                                (XrdSfsXferSize)aiop->sfsAio.aio_nbytes);
   aiop->doneWrite();
   return 0;
}

/******************************************************************************/
/*                               g e t M m a p                                */
/******************************************************************************/

int XrdBwmFile::getMmap(void **Addr, off_t &Size)         // Out
/*
  Function: Return memory mapping for file, if any.

  Output:   Addr        - Address of memory location
            Size        - Size of the file or zero if not memory mapped.
            Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{

// Mapping is not supported
//
   *Addr= 0;
   Size = 0;

   return SFS_OK;
}
  
/******************************************************************************/
/*                                  s t a t                                   */
/******************************************************************************/

int XrdBwmFile::stat(struct stat     *buf)         // Out
/*
  Function: Return file status information

  Input:    buf         - The stat structiure to hold the results

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   EPNAME("fstat");
   static unsigned int myInode = 0;
   union {long long   Fill;
          int         Xor[2];
          XrdBwmFile *fP;
          dev_t       Num;
         } theDev;

// Perform any required tracing
//
   FTRACE(calls, FName());

// Develop the device number
//
   theDev.Fill = 0; theDev.fP = this; theDev.Xor[0] ^= theDev.Xor[1];

// Fill out the stat structure for this pseudo file
//
   memset(buf, 0, sizeof(struct stat));
   buf->st_ino = myInode++;
   buf->st_dev = theDev.Num;
   buf->st_blksize = 4096;
   buf->st_mode = S_IFBLK;
   return SFS_OK;
}

/******************************************************************************/
/*                                  s y n c                                   */
/******************************************************************************/

int XrdBwmFile::sync()  // In
/*
  Function: Commit all unwritten bytes to physical media.

  Input:    n/a

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   EPNAME("sync");

// Perform any required tracing
//
   FTRACE(calls,"");

// We always succeed
//
   return SFS_OK;
}

/******************************************************************************/
/*                              s y n c   A I O                               */
/******************************************************************************/
  
// For now, reverts to synchronous case
//
int XrdBwmFile::sync(XrdSfsAio *aiop)
{
   aiop->Result = this->sync();
   aiop->doneWrite();
   return 0;
}

/******************************************************************************/
/*                              t r u n c a t e                               */
/******************************************************************************/

int XrdBwmFile::truncate(XrdSfsFileOffset  flen)  // In
/*
  Function: Set the length of the file object to 'flen' bytes.

  Input:    flen      - The new size of the file.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.

  Notes: 1. Truncate is not supported.
*/
{
   EPNAME("trunc");

// Lock the file handle and perform any tracing
//
   FTRACE(calls, "len=" <<flen);

// Return an error
//
   return  XrdBwmFS.Emsg("trunc", error, ENOTSUP, "truncate", oh->Name());
}

/******************************************************************************/
/*                             g e t C X i n f o                              */
/******************************************************************************/
  
int XrdBwmFile::getCXinfo(char cxtype[4], int &cxrsz)
/*
  Function: Set the length of the file object to 'flen' bytes.

  Input:    n/a

  Output:   cxtype - Compression algorithm code
            cxrsz  - Compression region size

            Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{

// Indicate not compressed
//
   cxrsz = 0;
   cxtype[0] = cxtype[1] = cxtype[2] = cxtype[3] = 0;
   return SFS_OK;
}

/******************************************************************************/
/*                                                                            */
/*         F i l e   S y s t e m   O b j e c t   I n t e r f a c e s          */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/
/*                                 c h m o d                                  */
/******************************************************************************/

int XrdBwm::chmod(const char             *path,    // In
                        XrdSfsMode        Mode,    // In
                        XrdOucErrInfo    &einfo,   // Out
                  const XrdSecEntity     *client,  // In
                  const char             *info)    // In
/*
  Function: Change the mode on a file or directory.

  Input:    path      - Is the fully qualified name of the file to be removed.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
// Return an error
//
   return XrdBwmFS.Emsg("chmod", einfo, ENOTSUP, "change", path);
}

/******************************************************************************/
/*                                e x i s t s                                 */
/******************************************************************************/

int XrdBwm::exists(const char                *path,        // In
                         XrdSfsFileExistence &file_exists, // Out
                         XrdOucErrInfo       &einfo,       // Out
                   const XrdSecEntity        *client,      // In
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
            info        - Opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.

  Notes:    When failure occurs, 'file_exists' is not modified.
*/
{

   file_exists=XrdSfsFileExistNo;
   return SFS_OK;
}

/******************************************************************************/
/*                                 f s c t l                                  */
/******************************************************************************/

int XrdBwm::fsctl(const int               cmd,
                  const char             *args,
                  XrdOucErrInfo          &einfo,
                  const XrdSecEntity     *client)
/*
  Function: Perform filesystem operations:

  Input:    cmd       - Operation command (currently supported):
                        None.
            arg       - Command dependent argument:
                      - STATXV: The file handle
            einfo     - Error/Response information structure.
            client    - Authentication credentials, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
// Operation is not supported
//
   return XrdBwmFS.Emsg("fsctl", einfo, ENOTSUP, "fsctl", args);
}

/******************************************************************************/
/*                            g e t V e r s i o n                             */
/******************************************************************************/
  
const char *XrdBwm::getVersion() {return XrdVSTRING;}

/******************************************************************************/
/*                                 m k d i r                                  */
/******************************************************************************/

int XrdBwm::mkdir(const char             *path,    // In
                        XrdSfsMode        Mode,    // In
                        XrdOucErrInfo    &einfo,   // Out
                  const XrdSecEntity     *client,  // In
                  const char             *info)    // In
/*
  Function: Create a directory entry.

  Input:    path      - Is the fully qualified name of the file to be removed.
            Mode      - Is the POSIX mode value the directory is to have.
                        Additionally, Mode may contain SFS_O_MKPTH if the
                        full dircectory path should be created.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
// Return an error
//
   return XrdBwmFS.Emsg("mkdir", einfo, ENOTSUP, "mkdir", path);
}

/******************************************************************************/
/*                               p r e p a r e                                */
/******************************************************************************/

int XrdBwm::prepare(      XrdSfsPrep       &pargs,      // In
                          XrdOucErrInfo    &out_error,  // Out
                    const XrdSecEntity     *client)     // In
{
   return 0;
}
  
/******************************************************************************/
/*                                r e m o v e                                 */
/******************************************************************************/

int XrdBwm::remove(const char              type,    // In
                   const char             *path,    // In
                         XrdOucErrInfo    &einfo,   // Out
                   const XrdSecEntity     *client,  // In
                   const char             *info)    // In
/*
  Function: Delete a file from the namespace and release it's data storage.

  Input:    type      - 'f' for file and 'd' for directory.
            path      - Is the fully qualified name of the file to be removed.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
// Return an error
//
   return XrdBwmFS.Emsg("remove", einfo, ENOTSUP, "remove", path);
}

/******************************************************************************/
/*                                r e n a m e                                 */
/******************************************************************************/

int XrdBwm::rename(const char             *old_name,  // In
                   const char             *new_name,  // In
                         XrdOucErrInfo    &einfo,     //Out
                   const XrdSecEntity     *client,    // In
                   const char             *infoO,     // In
                   const char             *infoN)     // In
/*
  Function: Renames a file with name 'old_name' to 'new_name'.

  Input:    old_name  - Is the fully qualified name of the file to be renamed.
            new_name  - Is the fully qualified name that the file is to have.
            einfo     - Error information structure, if an error occurs.
            client    - Authentication credentials, if any.
            infoO     - old_name opaque information to be used as seen fit.
            infoN     - new_name opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
// Return an error
//
   return XrdBwmFS.Emsg("rename", einfo, ENOTSUP, "rename", old_name);
}

/******************************************************************************/
/*                                  s t a t                                   */
/******************************************************************************/

int XrdBwm::stat(const char             *path,        // In
                       struct stat      *buf,         // Out
                       XrdOucErrInfo    &einfo,       // Out
                 const XrdSecEntity     *client,      // In
                 const char             *info)        // In
/*
  Function: Return file status information

  Input:    path      - The path for which status is wanted
            buf       - The stat structure to hold the results
            einfo     - Error information structure, if an error occurs.
            client    - Authentication credentials, if any.
            info      - opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
// Return an error
//
   return XrdBwmFS.Emsg("stat", einfo, ENOTSUP, "locate", path);
}

/******************************************************************************/

int XrdBwm::stat(const char             *path,        // In
                       mode_t           &mode,        // Out
                       XrdOucErrInfo    &einfo,       // Out
                 const XrdSecEntity     *client,      // In
                 const char             *info)        // In
/*
  Function: Return file status information (resident files only)

  Input:    path      - The path for which status is wanted
            mode      - The stat mode entry (faked -- do not trust it)
            einfo     - Error information structure, if an error occurs.
            client    - Authentication credentials, if any.
            info      - opaque information to be used as seen fit.

  Output:   Always returns SFS_ERROR if a delay needs to be imposed. Otherwise,
            SFS_OK is returned and mode is appropriately, if inaccurately, set.
            If file residency cannot be determined, mode is set to -1.
*/
{
// Return an error
//
   return XrdBwmFS.Emsg("stat", einfo, ENOTSUP, "locate", path);
}

/******************************************************************************/
/*                              t r u n c a t e                               */
/******************************************************************************/

int XrdBwm::truncate(const char             *path,    // In
                           XrdSfsFileOffset  Size,    // In
                           XrdOucErrInfo    &einfo,   // Out
                     const XrdSecEntity     *client,  // In
                     const char             *info)    // In
/*
  Function: Change the mode on a file or directory.

  Input:    path      - Is the fully qualified name of the file to be removed.
            Size      - the size the file should have.
            einfo     - Error information object to hold error details.
            client    - Authentication credentials, if any.
            info      - Opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
// Return an error
//
   return XrdBwmFS.Emsg("truncate", einfo, ENOTSUP, "truncate", path);
}

/******************************************************************************/
/*                                  E m s g                                   */
/******************************************************************************/

int XrdBwm::Emsg(const char    *pfx,    // Message prefix value
                 XrdOucErrInfo &einfo,  // Place to put text & error code
                 int            ecode,  // The error code
                 const char    *op,     // Operation being performed
                 const char    *target) // The target (e.g., fname)
{
   char *etext, buffer[MAXPATHLEN+80], unkbuff[64];

// Get the reason for the error
//
   if (ecode < 0) ecode = -ecode;
   if (!(etext = BwmEroute.ec2text(ecode))) 
      {sprintf(unkbuff, "reason unknown (%d)", ecode); etext = unkbuff;}

// Format the error message
//
   snprintf(buffer,sizeof(buffer),"Unable to %s %s; %s", op, target, etext);

// Print it out if debugging is enabled
//
#ifndef NODEBUG
   BwmEroute.Emsg(pfx, einfo.getErrUser(), buffer);
#endif

// Place the error message in the error object and return
//
   einfo.setErrInfo(ecode, buffer);
   return SFS_ERROR;
}

/******************************************************************************/

int XrdBwm::Emsg(const char    *pfx,    // Message prefix value
                 XrdOucErrInfo &einfo,  // Place to put text & error code
                 const char    *item,   // What is missing
                 const char    *op,     // Operation being performed
                 const char    *target) // The target (e.g., fname)
{
   char buffer[MAXPATHLEN+80];

// Format the error message
//
   snprintf(buffer,sizeof(buffer),"Unable to %s %s; %s missing",
                                   op, target, item);

// Print it out if debugging is enabled
//
#ifndef NODEBUG
   BwmEroute.Emsg(pfx, einfo.getErrUser(), buffer);
#endif

// Place the error message in the error object and return
//
   einfo.setErrInfo(EINVAL, buffer);
   return SFS_ERROR;
}

/******************************************************************************/
/*                                 S t a l l                                  */
/******************************************************************************/
  
int XrdBwm::Stall(XrdOucErrInfo   &einfo, // Error text & code
                  int              stime, // Seconds to stall
                  const char      *path)  // The path to stall on
{
    EPNAME("Stall")
#ifndef NODEBUG
    const char *tident = einfo.getErrUser();
#endif

// Trace the stall
//
   ZTRACE(delay, "Stall " <<stime <<" for " <<path);

// Place the error message in the error object and return
//
   einfo.setErrInfo(0, "");
   return stime;
}
