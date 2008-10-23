/******************************************************************************/
/*                                                                            */
/*                             X r d O f s . c c                              */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOfsCVSID = "$Id$";

/* Available compile-time define symbols:

   -DAIX       mangles some includes to accomodate AIX.

   -DNODEBUG   suppresses inline dbugging statement.

   -DNOSEC     suppresses security code generation.

   Note: This is a C++ mt-safe 64-bit clean program and must be compiled with:

         Solaris: -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 -D_REENTRANT

         AIX:     -D_THREAD_SAFE
*/

#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <memory.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <netdb.h>
#include <stdlib.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>

#include "XrdVersion.hh"

#include "XrdOfs/XrdOfs.hh"
#include "XrdOfs/XrdOfsEvs.hh"
#include "XrdOfs/XrdOfsTrace.hh"
#include "XrdOfs/XrdOfsSecurity.hh"

#include "XrdCms/XrdCmsClient.hh"

#include "XrdOss/XrdOss.hh"

#include "XrdNet/XrdNetDNS.hh"

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucLock.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucTrace.hh"
#include "XrdSec/XrdSecEntity.hh"
#include "XrdSfs/XrdSfsAio.hh"
#include "XrdSfs/XrdSfsInterface.hh"

#ifdef AIX
#include <sys/mode.h>
#endif

/******************************************************************************/
/*                  E r r o r   R o u t i n g   O b j e c t                   */
/******************************************************************************/

XrdSysError      OfsEroute(0);

XrdOucTrace      OfsTrace(&OfsEroute);

/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/
  
XrdOfsHandle     *XrdOfs::dummyHandle;

int               XrdOfs::MaxDelay = 60;
int               XrdOfs::OSSDelay = 30;

/******************************************************************************/
/*                    F i l e   S y s t e m   O b j e c t                     */
/******************************************************************************/
  
extern XrdOfs XrdOfsFS;

/******************************************************************************/
/*                 S t o r a g e   S y s t e m   O b j e c t                  */
/******************************************************************************/
  
XrdOss *XrdOfsOss;

/******************************************************************************/
/*                    X r d O f s   C o n s t r u c t o r                     */
/******************************************************************************/

XrdOfs::XrdOfs()
{
   unsigned int myIPaddr = 0;
   char buff[256], *bp;
   int i;

// Establish defaults
//
   Authorization = 0;
   Finder        = 0;
   Balancer      = 0;
   evsObject     = 0;
   myRole        = strdup("server");

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

// Set the configuration file name abd dummy handle
//
   ConfigFN = 0;
   XrdOfsHandle::Alloc(&dummyHandle);
}
  
/******************************************************************************/
/*                X r d O f s F i l e   C o n s t r u c t o r                 */
/******************************************************************************/

XrdOfsFile::XrdOfsFile(const char *user) : XrdSfsFile(user)
{
   oh = XrdOfs::dummyHandle; 
   dorawio = 0;
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
   OfsEroute.SetPrefix("ofs_");
   OfsEroute.logger(lp);
   OfsEroute.Say("Copr.  2008 Stanford University, Ofs Version " XrdVSTRING);

// Initialize the subsystems
//
   XrdOfsFS.ConfigFN = (configfn && *configfn ? strdup(configfn) : 0);
   if ( XrdOfsFS.Configure(OfsEroute) ) return 0;

// All done, we can return the callout vector to these routines.
//
   return &XrdOfsFS;
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

int XrdOfsDirectory::open(const char              *dir_path, // In
                          const XrdSecEntity      *client,   // In
                          const char              *info)      // In
/*
  Function: Open the directory `path' and prepare for reading.

  Input:    path      - The fully qualified name of the directory to open.
            client    - Authentication credentials, if any.
            info      - Opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success, otherwise SFS_ERROR.

  Notes: 1. The code here assumes that directory file descriptors are never
            shared. Hence, no locks need to be obtained. It works out that
            lock overhead is worse than have a duplicate file descriptor for
            very short durations.
*/
{
   EPNAME("opendir");
   XrdOucEnv Open_Env(info);
   int retc;

// Trace entry
//
   XTRACE(opendir, dir_path, "");

// Verify that this object is not already associated with an open directory
//
   if (dp) return
      XrdOfsFS.Emsg(epname, error, EADDRINUSE, "open directory", dir_path);

// Apply security, as needed
//
   AUTHORIZE(client,&Open_Env,AOP_Readdir,"open directory",dir_path,error);

// Open the directory and allocate a handle for it
//
   if (!(dp = XrdOfsOss->newDir(tident))) retc = -ENOMEM;
      else if (!(retc = dp->Opendir(dir_path)))
              {fname = strdup(dir_path);
               return SFS_OK;
              }
              else {delete dp; dp = 0;}

// Encountered an error
//
   return XrdOfsFS.Emsg(epname, error, retc, "open directory", dir_path);
}

/******************************************************************************/
/*                             n e x t E n t r y                              */
/******************************************************************************/

const char *XrdOfsDirectory::nextEntry()
/*
  Function: Read the next directory entry.

  Input:    n/a

  Output:   Upon success, returns the contents of the next directory entry as
            a null terminated string. Returns a null pointer upon EOF or an
            error. To differentiate the two cases, getErrorInfo will return
            0 upon EOF and an actual error code (i.e., not 0) on error.

  Notes: 1. The code here assumes that idle directory file descriptors are
            *not* closed. This needs to be the case because we need to return
            non-duplicate directory entries. Anyway, the xrootd readdir protocol
            is handled internally so directories should never be idle.
         2. The code here assumes that directory file descriptors are never
            shared. Hence, no locks need to be obtained. It works out that
            lock overhead is worse than have a duplicate file descriptor for
            very short durations.
*/
{
   EPNAME("readdir");
   int retc;

// Check if this directory is actually open
//
   if (!dp) {XrdOfsFS.Emsg(epname, error, EBADF, "read directory");
             return 0;
            }

// Check if we are at EOF (once there we stay there)
//
   if (atEOF) return 0;

// Read the next directory entry
//
   if ((retc = dp->Readdir(dname, sizeof(dname))) < 0)
      {XrdOfsFS.Emsg(epname, error, retc, "read directory", fname);
       return 0;
      }

// Check if we have reached end of file
//
   if (!*dname)
      {atEOF = 1;
       error.clear();
       XTRACE(readdir, fname, "<eof>");
       return 0;
      }

// Return the actual entry
//
   XTRACE(readdir, fname, dname);
   return (const char *)(dname);
}

/******************************************************************************/
/*                                 c l o s e                                  */
/******************************************************************************/
  
int XrdOfsDirectory::close()
/*
  Function: Close the directory object.

  Input:    n/a

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.

  Notes: 1. The code here assumes that directory file descriptors are never
            shared. Hence, no locks need to be obtained. It works out that
            lock overhead is worse than have a duplicate file descriptor for
            very short durations.
*/
{
   EPNAME("closedir");
   int retc;

// Check if this directory is actually open
//
   if (!dp) {XrdOfsFS.Emsg(epname, error, EBADF, "close directory");
             return SFS_ERROR;
            }
   XTRACE(closedir, fname, "");

// Close this directory
//
    if ((retc = dp->Close()))
       retc = XrdOfsFS.Emsg(epname, error, retc, "close", fname);
       else retc = SFS_OK;

// All done
//
   delete dp;
   dp = 0;
   free(fname);
   fname = 0;
   return retc;
}

/******************************************************************************/
/*                                                                            */
/*                F i l e   O b j e c t   I n t e r f a c e s                 */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/
/*                                  o p e n                                   */
/******************************************************************************/

int XrdOfsFile::open(const char          *path,      // In
                     XrdSfsFileOpenMode   open_mode, // In
                     mode_t               Mode,      // In
               const XrdSecEntity        *client,    // In
               const char                *info)      // In
/*
  Function: Open the file `path' in the mode indicated by `open_mode'.  

  Input:    path      - The fully qualified name of the file to open.
            open_mode - One of the following flag values:
                        SFS_O_RDONLY - Open file for reading.
                        SFS_O_WRONLY - Open file for writing.
                        SFS_O_RDWR   - Open file for update
                        SFS_O_CREAT  - Create the file open in RW mode
                        SFS_O_TRUNC  - Trunc  the file open in RW mode
            Mode      - The Posix access mode bits to be assigned to the file.
                        These bits correspond to the standard Unix permission
                        bits (e.g., 744 == "rwxr--r--"). Additionally, Mode
                        may contain SFS_O_MKPTH to force creation of the full
                        directory path if it does not exist. This parameter is
                        ignored unless open_mode = SFS_O_CREAT.
            client    - Authentication credentials, if any.
            info      - Opaque information to be used as seen fit.

  Output:   Returns SFS_OK upon success, otherwise SFS_ERROR is returned.
*/
{
   EPNAME("open");
   XrdOfsHandle *hP;
   int retc, isRW, find_flag, open_flag = 0;
   int crOpts = (Mode & SFS_O_MKPTH ? XRDOSS_mkpath : 0);
   XrdOssDF           *fp;
   XrdOucEnv Open_Env(info);

// Trace entry
//
   ZTRACE(open, std::hex <<open_mode <<"-" <<std::oct <<Mode <<std::dec <<" fn=" <<path);

// Verify that this object is not already associated with an open file
//
   XrdOfsFS.ocMutex.Lock();
   if (oh != XrdOfs::dummyHandle)
      {XrdOfsFS.ocMutex.UnLock();
       return XrdOfsFS.Emsg(epname,error,EADDRINUSE,"open file",path);
      }
   XrdOfsFS.ocMutex.UnLock();

// Set the actual open mode and find mode
//
   find_flag = open_mode & (SFS_O_NOWAIT | SFS_O_RESET);
   if (open_mode & SFS_O_CREAT) open_mode = SFS_O_CREAT;
      else if (open_mode & SFS_O_TRUNC) open_mode = SFS_O_TRUNC;

   switch(open_mode & (SFS_O_RDONLY | SFS_O_WRONLY | SFS_O_RDWR |
                       SFS_O_CREAT  | SFS_O_TRUNC))
   {
   case SFS_O_CREAT:  open_flag   = O_RDWR     | O_CREAT  | O_EXCL;
                      find_flag  |= SFS_O_RDWR | SFS_O_CREAT;
                      crOpts     |= XRDOSS_new;
                      isRW = 1;
                      break;
   case SFS_O_TRUNC:  open_flag  |= O_RDWR     | O_CREAT     | O_TRUNC;
                      find_flag  |= SFS_O_RDWR | SFS_O_TRUNC;
                      isRW = 1;
                      break;
   case SFS_O_RDONLY: open_flag = O_RDONLY; find_flag |= SFS_O_RDONLY;
                      isRW = 0;
                      break;
   case SFS_O_WRONLY: open_flag = O_WRONLY; find_flag |= SFS_O_WRONLY;
                      isRW = 1;
                      break;
   case SFS_O_RDWR:   open_flag = O_RDWR;   find_flag |= SFS_O_RDWR;
                      isRW = 1;
                      break;
   default:           open_flag = O_RDONLY; find_flag |= SFS_O_RDONLY;
                      isRW = 0;
                      break;
   }


// If we have a finder object, use it to direct the client. The final
// destination will apply the security that is needed
//
   if (XrdOfsFS.Finder && (retc = XrdOfsFS.Finder->Locate(error, path,
                                                   find_flag, &Open_Env)))
      return XrdOfsFS.fsError(error, retc);

// Create the file if so requested o/w try to attach the file
//
   if (open_flag & O_CREAT)
      {// Apply security, as needed
       //
       AUTHORIZE(client,&Open_Env,AOP_Create,"create",path,error);
       OOIDENTENV(client, Open_Env);

       // Create the file. If ENOTSUP is returned, promote the creation to
       // the subsequent open. This is to accomodate proxy support.
       //
       if ((retc = XrdOfsOss->Create(tident, path, Mode & S_IAMB, Open_Env,
                                     ((open_flag << 8) | crOpts))))
          {if (retc > 0) return XrdOfsFS.Stall(error, retc, path);
           if (retc == -EINPROGRESS)
              {XrdOfsFS.evrObject.Wait4Event(path,&error);
               return XrdOfsFS.fsError(error, retc);
              }
           if (retc != -ENOTSUP)
              return XrdOfsFS.Emsg(epname, error, retc, "create", path);
          } else {
            if (XrdOfsFS.Balancer) XrdOfsFS.Balancer->Added(path);
            open_flag  = O_RDWR|O_TRUNC;
            if (XrdOfsFS.evsObject 
            &&  XrdOfsFS.evsObject->Enabled(XrdOfsEvs::Create))
               {XrdOfsEvsInfo evInfo(tident,path,info,&Open_Env,Mode);
                XrdOfsFS.evsObject->Notify(XrdOfsEvs::Create, evInfo);
               }
          }

      } else {

       // Apply security, as needed
       //
       AUTHORIZE(client,&Open_Env,(isRW?AOP_Update:AOP_Read),"open",path,error);
       OOIDENTENV(client, Open_Env);
      }

// Get a handle for this file.
//
   if ((retc = XrdOfsHandle::Alloc(path, isRW, &hP)))
      {if (retc > 0) return XrdOfsFS.Stall(error, retc, path);
       return XrdOfsFS.Emsg(epname, error, retc, "attach", path);
      }

// If this is a previously existing handle, we are done
//
   if (!(hP->Inactive()))
      {dorawio = (oh->isCompressed && open_mode & SFS_O_RAWIO ? 1 : 0);
       XrdOfsFS.ocMutex.Lock(); oh = hP; XrdOfsFS.ocMutex.UnLock();
       FTRACE(open, "attach use=" <<oh->Usage());
       hP->UnLock();
       return SFS_OK;
      }

// Get a storage system object
//
   if (!(fp = XrdOfsOss->newFile(tident)))
      {hP->Retire();
       return XrdOfsFS.Emsg(epname, error, ENOMEM, "open", path);
      }

// Open the file
//
   if ((retc = fp->Open(path, open_flag, Mode, Open_Env)))
      {hP->Retire(); delete fp;
       if (retc > 0) return XrdOfsFS.Stall(error, retc, path);
       if (retc == -EINPROGRESS)
          {XrdOfsFS.evrObject.Wait4Event(path,&error);
           return XrdOfsFS.fsError(error, retc);
          }
       return XrdOfsFS.Emsg(epname, error, retc, "open", path);
      }

// Set compression values and activate the handle
//
   if (fp->isCompressed()) 
      {hP->isCompressed = 1;
       dorawio = (open_mode & SFS_O_RAWIO ? 1 : 0);
      }
   hP->Activate(fp);
   hP->UnLock();

// Send an open event if we must
//
   if (XrdOfsFS.evsObject)
      {XrdOfsEvs::Event theEvent = (hP->isRW?XrdOfsEvs::Openw:XrdOfsEvs::Openr);
       if (XrdOfsFS.evsObject->Enabled(theEvent))
          {XrdOfsEvsInfo evInfo(tident, path, info, &Open_Env);
           XrdOfsFS.evsObject->Notify(theEvent, evInfo);
          }
      }

// All done
//
   XrdOfsFS.ocMutex.Lock(); oh = hP; XrdOfsFS.ocMutex.UnLock();
   return SFS_OK;
}

/******************************************************************************/
/*                                 c l o s e                                  */
/******************************************************************************/

int XrdOfsFile::close()  // In
/*
  Function: Close the file object.

  Input:    n/a

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   EPNAME("close");
   XrdOfsHandle *hP;

// Trace the call
//
    FTRACE(close, "use=" <<oh->Usage()); // Unreliable trace, no origin lock

// Verify the handle (we briefly maintain a global lock)
//
    XrdOfsFS.ocMutex.Lock();
    if (oh == XrdOfs::dummyHandle)
       {XrdOfsFS.ocMutex.UnLock(); return SFS_OK;}
    if ((oh->Inactive()))
       {XrdOfsFS.ocMutex.UnLock();
        return XrdOfsFS.Emsg(epname, error, EBADF, "close file");
       }
    hP = oh; oh = XrdOfs::dummyHandle;
    XrdOfsFS.ocMutex.UnLock();
    hP->Lock();

// We need to handle the cunudrum that an event may have to be sent upon
// the final close. However, that would cause the path name to be destroyed.
// So, we have two modes of logic where we copy out the pathname if a final
// close actually occurs. The path is not copied if it's not final and we
// don't bother with any of it if we need not generate an event.
//
   if (XrdOfsFS.evsObject && tident
   &&  XrdOfsFS.evsObject->Enabled(hP->isRW ? XrdOfsEvs::Closew
                                            : XrdOfsEvs::Closer))
      {long long FSize, *retsz;
       char pathbuff[MAXPATHLEN+8];
       XrdOfsEvs::Event theEvent;
       if (hP->isRW) {theEvent = XrdOfsEvs::Closew; retsz = &FSize;}
          else {      theEvent = XrdOfsEvs::Closer; retsz = 0; FSize=0;}
       if (!(hP->Retire(retsz, pathbuff, sizeof(pathbuff))))
          {XrdOfsEvsInfo evInfo(tident, pathbuff, "" , 0, 0, FSize);
           XrdOfsFS.evsObject->Notify(theEvent, evInfo);
          } else hP->Retire();
      } else     hP->Retire();

// All done
//
    return SFS_OK;
}

/******************************************************************************/
/*                                  f c t l                                   */
/******************************************************************************/
  
int            XrdOfsFile::fctl(const int               cmd,
                                const char             *args,
                                      XrdOucErrInfo    &out_error)
{
// See if we can do this
//
   if (cmd == SFS_FCTL_GETFD)
      {out_error.setErrCode(oh->Select().getFD());
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

int            XrdOfsFile::read(XrdSfsFileOffset  offset,    // In
                                XrdSfsXferSize    blen)      // In
/*
  Function: Preread `blen' bytes at `offset'

  Input:    offset    - The absolute byte offset at which to start the read.
            blen      - The amount to preread.

  Output:   Returns SFS_OK upon success and SFS_ERROR o/w.
*/
{
   EPNAME("read");
   int retc;

// Perform required tracing
//
   FTRACE(read, "preread " <<blen <<"@" <<offset);

// Make sure the offset is not too large
//
#if _FILE_OFFSET_BITS!=64
   if (offset >  0x000000007fffffff)
      return  XrdOfsFS.Emsg(epname, error, EFBIG, "read", oh->Name());
#endif

// Now preread the actual number of bytes
//
   if ((retc = oh->Select().Read((off_t)offset, (size_t)blen)) < 0)
      return XrdOfsFS.Emsg(epname, error, (int)retc, "preread", oh->Name());

// Return number of bytes read
//
   return retc;
}
  
/******************************************************************************/
/*                                  r e a d                                   */
/******************************************************************************/

XrdSfsXferSize XrdOfsFile::read(XrdSfsFileOffset  offset,    // In
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
   EPNAME("read");
   XrdSfsXferSize nbytes;

// Perform required tracing
//
   FTRACE(read, blen <<"@" <<offset);

// Make sure the offset is not too large
//
#if _FILE_OFFSET_BITS!=64
   if (offset >  0x000000007fffffff)
      return  XrdOfsFS.Emsg(epname, error, EFBIG, "read", oh->Name());
#endif

// Now read the actual number of bytes
//
   nbytes = (dorawio ?
            (XrdSfsXferSize)(oh->Select().ReadRaw((void *)buff,
                            (off_t)offset, (size_t)blen))
          : (XrdSfsXferSize)(oh->Select().Read((void *)buff,
                            (off_t)offset, (size_t)blen)));
   if (nbytes < 0)
      return XrdOfsFS.Emsg(epname, error, (int)nbytes, "read", oh->Name());

// Return number of bytes read
//
   return nbytes;
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

int XrdOfsFile::read(XrdSfsAio *aiop)
{
   EPNAME("aioread");
   int rc;

// Async mode for compressed files is not supported.
//
   if (oh->isCompressed)
      {aiop->Result = this->read((XrdSfsFileOffset)aiop->sfsAio.aio_offset,
                                           (char *)aiop->sfsAio.aio_buf,
                                   (XrdSfsXferSize)aiop->sfsAio.aio_nbytes);
       aiop->doneRead();
       return 0;
      }

// Perform required tracing
//
   FTRACE(aio, aiop->sfsAio.aio_nbytes <<"@" <<aiop->sfsAio.aio_offset);

// Make sure the offset is not too large
//
#if _FILE_OFFSET_BITS!=64
   if (aiop->sfsAio.aio_offset >  0x000000007fffffff)
      return  XrdOfsFS.Emsg(epname, error, EFBIG, "read", oh->Name());
#endif

// Issue the read. Only true errors are returned here.
//
   if ((rc = oh->Select().Read(aiop)) < 0)
      return XrdOfsFS.Emsg(epname, error, rc, "read", oh->Name());

// All done
//
   return SFS_OK;
}

/******************************************************************************/
/*                                 w r i t e                                  */
/******************************************************************************/

XrdSfsXferSize XrdOfsFile::write(XrdSfsFileOffset  offset,    // In
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

  Notes:    An error return may be delayed until the next write(), close(), or
            sync() call.
*/
{
   EPNAME("write");
   XrdSfsXferSize nbytes;

// Perform any required tracing
//
   FTRACE(write, blen <<"@" <<offset);

// Make sure the offset is not too large
//
#if _FILE_OFFSET_BITS!=64
   if (offset >  0x000000007fffffff)
      return  XrdOfsFS.Emsg(epname, error, EFBIG, "read", oh->Name());
#endif

// Silly Castor stuff
//
   if (XrdOfsFS.evsObject && !(oh->isChanged)
   &&  XrdOfsFS.evsObject->Enabled(XrdOfsEvs::Fwrite)) GenFWEvent();

// Write the requested bytes
//
   oh->isPending = 1;
   nbytes = (XrdSfsXferSize)(oh->Select().Write((const void *)buff,
                            (off_t)offset, (size_t)blen));
   if (nbytes < 0)
      return XrdOfsFS.Emsg(epname, error, (int)nbytes, "write", oh->Name());

// Return number of bytes written
//
   return nbytes;
}

/******************************************************************************/
/*                             w r i t e   A I O                              */
/******************************************************************************/
  
// For now, this reverts to synchronous I/O
//
int XrdOfsFile::write(XrdSfsAio *aiop)
{
   EPNAME("aiowrite");
   int rc;

// Perform any required tracing
//
   FTRACE(aio, aiop->sfsAio.aio_nbytes <<"@" <<aiop->sfsAio.aio_offset);

// Make sure the offset is not too large
//
#if _FILE_OFFSET_BITS!=64
   if (aiop->sfsAio.aio_offset >  0x000000007fffffff)
      return  XrdOfsFS.Emsg(epname, error, EFBIG, "read", oh->Name());
#endif

// Silly Castor stuff
//
   if (XrdOfsFS.evsObject && !(oh->isChanged)
   &&  XrdOfsFS.evsObject->Enabled(XrdOfsEvs::Fwrite)) GenFWEvent();

// Write the requested bytes
//
   oh->isPending = 1;
   if ((rc = oh->Select().Write(aiop)) < 0)
      return XrdOfsFS.Emsg(epname, error, rc, "write", oh->Name());

// All done
//
   return SFS_OK;
}

/******************************************************************************/
/*                               g e t M m a p                                */
/******************************************************************************/

int XrdOfsFile::getMmap(void **Addr, off_t &Size)         // Out
/*
  Function: Return memory mapping for file, if any.

  Output:   Addr        - Address of memory location
            Size        - Size of the file or zero if not memory mapped.
            Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{

// Perform the function
//
   Size = oh->Select().getMmap(Addr);

   return SFS_OK;
}
  
/******************************************************************************/
/*                                  s t a t                                   */
/******************************************************************************/

int XrdOfsFile::stat(struct stat     *buf)         // Out
/*
  Function: Return file status information

  Input:    buf         - The stat structiure to hold the results

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   EPNAME("fstat");
   int retc;

// Lock the handle and perform any required tracing
//
   FTRACE(stat, "");

// Perform the function
//
   if ((retc = oh->Select().Fstat(buf)) < 0)
      return XrdOfsFS.Emsg(epname,error,retc,"get state for",oh->Name());

   return SFS_OK;
}

/******************************************************************************/
/*                                  s y n c                                   */
/******************************************************************************/

int XrdOfsFile::sync()  // In
/*
  Function: Commit all unwritten bytes to physical media.

  Input:    n/a

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   EPNAME("sync");
   int retc;

// Perform any required tracing
//
   FTRACE(sync, "");

// We can test the pendio flag w/o a lock because the person doing this
// sync must have done the previous write. Causality is the synchronizer.
//
   if (!(oh->isPending)) return SFS_OK;

// We can also skip the sync if the file is closed. However, we need a file
// object lock in order to test the flag. We can also reset the PENDIO flag.
//
   oh->Lock();
   oh->isPending = 0;
   oh->UnLock();

// Perform the function
//
   if ((retc = oh->Select().Fsync()))
      {oh->isPending = 1;
       return XrdOfsFS.Emsg(epname, error, retc, "synchronize", oh->Name());
      }

// Indicate all went well
//
   return SFS_OK;
}


/******************************************************************************/
/*                              s y n c   A I O                               */
/******************************************************************************/
  
// For now, reverts to synchronous case
//
int XrdOfsFile::sync(XrdSfsAio *aiop)
{
   aiop->Result = this->sync();
   aiop->doneWrite();
   return 0;
}

/******************************************************************************/
/*                              t r u n c a t e                               */
/******************************************************************************/

int XrdOfsFile::truncate(XrdSfsFileOffset  flen)  // In
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
   EPNAME("trunc");
   int retc;

// Lock the file handle and perform any tracing
//
   FTRACE(truncate, "len=" <<flen);

// Make sure the offset is not too large
//
   if (sizeof(off_t) < sizeof(flen) && flen >  0x000000007fffffff)
      return  XrdOfsFS.Emsg(epname, error, EFBIG, "truncate", oh->Name());

// Silly Castor stuff
//
   if (XrdOfsFS.evsObject && !(oh->isChanged)
   &&  XrdOfsFS.evsObject->Enabled(XrdOfsEvs::Fwrite)) GenFWEvent();

// Perform the function
//
   oh->isPending = 1;
   if ((retc = oh->Select().Ftruncate(flen)))
      return XrdOfsFS.Emsg(epname, error, retc, "truncate", oh->Name());

// Indicate Success
//
   return SFS_OK;
}

/******************************************************************************/
/*                             g e t C X i n f o                              */
/******************************************************************************/
  
int XrdOfsFile::getCXinfo(char cxtype[4], int &cxrsz)
/*
  Function: Set the length of the file object to 'flen' bytes.

  Input:    n/a

  Output:   cxtype - Compression algorithm code
            cxrsz  - Compression region size

            Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{

// Copy out the info
//
   cxrsz = oh->Select().isCompressed(cxtype);
   return SFS_OK;
}

/******************************************************************************/
/*                  P r i v a t e   F i l e   M e t h o d s                   */
/******************************************************************************/
/******************************************************************************/
/* protected                  G e n F W E v e n t                             */
/******************************************************************************/
  
void XrdOfsFile::GenFWEvent()
{
   int first_write;

// This silly code is to generate a 1st write event which slows things down
// but is needed by the one and only Castor. What a big sigh!
//
   oh->Lock();
   if ((first_write = !(oh->isChanged))) oh->isChanged = 1;
   oh->UnLock();
   if (first_write)
      {XrdOfsEvsInfo evInfo(tident, oh->Name());
       XrdOfsFS.evsObject->Notify(XrdOfsEvs::Fwrite, evInfo);
      }
}

/******************************************************************************/
/*                                                                            */
/*         F i l e   S y s t e m   O b j e c t   I n t e r f a c e s          */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/
/*                                 c h m o d                                  */
/******************************************************************************/

int XrdOfs::chmod(const char             *path,    // In
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
   EPNAME("chmod");
   mode_t acc_mode = Mode & S_IAMB;
   const char *tident = einfo.getErrUser();
   XrdOucEnv chmod_Env(info);
   int retc;
   XTRACE(chmod, path, "");

// Apply security, as needed
//
   AUTHORIZE(client,&chmod_Env,AOP_Chmod,"chmod",path,einfo);

// Find out where we should chmod this file
//
   if (Finder && Finder->isRemote())
      {if (fwdCHMOD.Cmd)
          {char buff[8];
           sprintf(buff, "%o", acc_mode);
           if (Forward(retc, einfo, fwdCHMOD, path, buff, info)) return retc;
          }
          else if ((retc = Finder->Locate(einfo,path,SFS_O_RDWR|SFS_O_META)))
                  return fsError(einfo, retc);
      }

// Check if we should generate an event
//
   if (evsObject && evsObject->Enabled(XrdOfsEvs::Chmod))
      {XrdOfsEvsInfo evInfo(tident, path, info, &chmod_Env, acc_mode);
       evsObject->Notify(XrdOfsEvs::Chmod, evInfo);
      }

// Now try to find the file or directory
//
   if (!(retc = XrdOfsOss->Chmod(path, acc_mode))) return SFS_OK;

// An error occured, return the error info
//
   return XrdOfsFS.Emsg(epname, einfo, retc, "change", path);
}

/******************************************************************************/
/*                                e x i s t s                                 */
/******************************************************************************/

int XrdOfs::exists(const char                *path,        // In
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
   EPNAME("exists");
   struct stat fstat;
   int retc;
   const char *tident = einfo.getErrUser();
   XrdOucEnv stat_Env(info);
   XTRACE(exists, path, "");

// Apply security, as needed
//
   AUTHORIZE(client,&stat_Env,AOP_Stat,"locate",path,einfo);

// Find out where we should stat this file
//
   if (Finder && Finder->isRemote() 
   &&  (retc = Finder->Locate(einfo, path, SFS_O_RDONLY)))
      return fsError(einfo, retc);

// Now try to find the file or directory
//
   retc = XrdOfsOss->Stat(path, &fstat);
   if (!retc)
      {     if (S_ISDIR(fstat.st_mode)) file_exists=XrdSfsFileExistIsDirectory;
       else if (S_ISREG(fstat.st_mode)) file_exists=XrdSfsFileExistIsFile;
       else                             file_exists=XrdSfsFileExistNo;
       return SFS_OK;
      }
   if (retc == -ENOENT)
      {file_exists=XrdSfsFileExistNo;
       return SFS_OK;
      }

// An error occured, return the error info
//
   return XrdOfsFS.Emsg(epname, einfo, retc, "locate", path);
}

/******************************************************************************/
/*                                 f s c t l                                  */
/******************************************************************************/

int XrdOfs::fsctl(const int               cmd,
                  const char             *args,
                  XrdOucErrInfo          &einfo,
                  const XrdSecEntity     *client)
/*
  Function: Perform filesystem operations:

  Input:    cmd       - Operation command (currently supported):
                        SFS_FSCTL_LOCATE - locate file
                        SFS_FSCTL_STATFS - return file system info (physical)
                        SFS_FSCTL_STATLS - return file system info (logical)
                        SFS_FSCTL_STATXA - return file extended attributes
            arg       - Command dependent argument:
                      - Locate: The path whose location is wanted
            buf       - The stat structure to hold the results
            einfo     - Error/Response information structure.
            client    - Authentication credentials, if any.

  Output:   Returns SFS_OK upon success and SFS_ERROR upon failure.
*/
{
   EPNAME("fsctl");
   static int PrivTab[]     = {XrdAccPriv_Delete, XrdAccPriv_Insert,
                               XrdAccPriv_Lock,   XrdAccPriv_Lookup,
                               XrdAccPriv_Rename, XrdAccPriv_Read,
                               XrdAccPriv_Write};
   static char PrivLet[]    = {'d',               'i',
                               'k',               'l',
                               'n',               'r',
                               'w'};
   static const int PrivNum = sizeof(PrivLet);

   int retc, find_flag = SFS_O_LOCATE | (cmd & (SFS_O_NOWAIT | SFS_O_RESET));
   int i, blen, privs, opcode = cmd & SFS_FSCTL_CMD;
   const char *tident = einfo.getErrUser();
   char *bP, *cP;
   XTRACE(fsctl, args, "");

// Process the LOCATE request
//
   if (opcode == SFS_FSCTL_LOCATE)
      {struct stat fstat;
       char rType[3], *Resp[] = {rType, locResp};
       AUTHORIZE(client,0,AOP_Stat,"locate",args,einfo);
       if (Finder && Finder->isRemote()
       &&  (retc = Finder->Locate(einfo, args, find_flag)))
          return fsError(einfo, retc);
       if ((retc = XrdOfsOss->Stat(args, &fstat)))
          return XrdOfsFS.Emsg(epname, einfo, retc, "locate", args);
       rType[0] = ((fstat.st_mode & S_IFBLK) == S_IFBLK ? 's' : 'S');
       rType[1] = (fstat.st_mode & S_IWUSR            ? 'w' : 'r');
       rType[2] = '\0';
       einfo.setErrInfo(locRlen+3, (const char **)Resp, 2);
       return SFS_DATA;
      }

// Process the STATFS request
//
   if (opcode == SFS_FSCTL_STATFS)
      {AUTHORIZE(client,0,AOP_Stat,"statfs",args,einfo);
       if (Finder && Finder->isRemote()
       &&  (retc = Finder->Space(einfo, args))) return fsError(einfo, retc);
       bP = einfo.getMsgBuff(blen);
       if ((retc = XrdOfsOss->StatFS(args, bP, blen)))
          return XrdOfsFS.Emsg(epname, einfo, retc, "statfs", args);
       einfo.setErrCode(blen+1);
       return SFS_DATA;
      }

// Process the STATLS request
//
   if (opcode == SFS_FSCTL_STATLS)
      {const char *path;
       char pbuff[1024], *opq = index(args, '?');
       XrdOucEnv statls_Env(opq ? opq+1 : 0);
       if (!opq) path = args;
          else {int plen = opq-args;
                if (plen >= (int)sizeof(pbuff)) plen = sizeof(pbuff)-1;
                strncpy(pbuff, args, plen);
                path = pbuff;
               }
       AUTHORIZE(client,0,AOP_Stat,"statfs",path,einfo);
       if (Finder && Finder->isRemote()
       &&  (retc = Finder->Space(einfo, path))) return fsError(einfo, retc);
       bP = einfo.getMsgBuff(blen);
       if ((retc = XrdOfsOss->StatLS(statls_Env, path, bP, blen)))
          return XrdOfsFS.Emsg(epname, einfo, retc, "statls", path);
       einfo.setErrCode(blen+1);
       return SFS_DATA;
      }

// Process the STATXA request
//
   if (opcode == SFS_FSCTL_STATXA)
      {AUTHORIZE(client,0,AOP_Stat,"statxa",args,einfo);
       if (Finder && Finder->isRemote()
       && (retc = Finder->Locate(einfo, args, SFS_O_RDONLY|SFS_O_STAT)))
          return fsError(einfo, retc);
       bP = einfo.getMsgBuff(blen);
       if ((retc = XrdOfsOss->StatXA(args, bP, blen)))
          return XrdOfsFS.Emsg(epname, einfo, retc, "statxa", args);
       if (!client || !XrdOfsFS.Authorization) privs = XrdAccPriv_All;
          else privs = XrdOfsFS.Authorization->Access(client, args, AOP_Any);
       cP = bP + blen; strcpy(cP, "&ofs.ap="); cP += 8;
       if (privs == XrdAccPriv_All) *cP++ = 'a';
          else {for (i = 0; i < PrivNum; i++)
                    if (PrivTab[i] & privs) *cP++ = PrivLet[i];
                if (cP == (bP + blen + 1)) *cP++ = '?';
               }
       *cP++ = '\0';
       einfo.setErrCode(cP-bP+1);
       return SFS_DATA;
      }

// Operation is not supported
//
   return XrdOfsFS.Emsg(epname, einfo, ENOTSUP, "fsctl", args);

}

/******************************************************************************/
/*                            g e t V e r s i o n                             */
/******************************************************************************/
  
const char *XrdOfs::getVersion() {return XrdVSTRING;}

/******************************************************************************/
/*                                 m k d i r                                  */
/******************************************************************************/

int XrdOfs::mkdir(const char             *path,    // In
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
   EPNAME("mkdir");
   static const int LocOpts = SFS_O_RDWR | SFS_O_CREAT | SFS_O_META;
   mode_t acc_mode = Mode & S_IAMB;
   int retc, mkpath = Mode & SFS_O_MKPTH;
   const char *tident = einfo.getErrUser();
   XrdOucEnv mkdir_Env(info);
   XTRACE(mkdir, path, "");

// Apply security, as needed
//
   AUTHORIZE(client,&mkdir_Env,AOP_Mkdir,"mkdir",path,einfo);

// Find out where we should remove this file
//
   if (Finder && Finder->isRemote())
      {if (fwdMKDIR.Cmd)
          {char buff[8];
           sprintf(buff, "%o", acc_mode);
           if (Forward(retc, einfo, (mkpath ? fwdMKPATH:fwdMKDIR),
                       path, buff, info)) return retc;
          }
          else if ((retc = Finder->Locate(einfo,path,LocOpts)))
                  return fsError(einfo, retc);
      }

// Perform the actual operation
//
    if ((retc = XrdOfsOss->Mkdir(path, acc_mode, mkpath)))
       return XrdOfsFS.Emsg(epname, einfo, retc, "mkdir", path);

// Check if we should generate an event
//
   if (evsObject && evsObject->Enabled(XrdOfsEvs::Mkdir))
      {XrdOfsEvsInfo evInfo(tident, path, info, &mkdir_Env, acc_mode);
       evsObject->Notify(XrdOfsEvs::Mkdir, evInfo);
      }

    return SFS_OK;
}

/******************************************************************************/
/*                               p r e p a r e                                */
/******************************************************************************/

int XrdOfs::prepare(      XrdSfsPrep       &pargs,      // In
                          XrdOucErrInfo    &out_error,  // Out
                    const XrdSecEntity     *client)     // In
{
   EPNAME("prepare");
   XrdOucTList *tp = pargs.paths;
   int retc;

// Run through the paths to make sure client can read each one
//
   while(tp)
        {AUTHORIZE(client,0,AOP_Read,"prepare",tp->text,out_error);
         tp = tp->next;
        }

// If we have a finder object, use it to prepare the paths. Otherwise,
// ignore this prepare request (we may change this in the future).
//
   if (XrdOfsFS.Finder && (retc = XrdOfsFS.Finder->Prepare(out_error, pargs)))
      return fsError(out_error, retc);
   return 0;
}
  
/******************************************************************************/
/*                                r e m o v e                                 */
/******************************************************************************/

int XrdOfs::remove(const char              type,    // In
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
   EPNAME("remove");
   int retc;
   const char *tident = einfo.getErrUser();
   XrdOucEnv rem_Env(info);
   XTRACE(remove, path, type);

// Apply security, as needed
//
   AUTHORIZE(client,&rem_Env,AOP_Delete,"remove",path,einfo);

// Find out where we should remove this file
//
   if (Finder && Finder->isRemote())
      {struct fwdOpt *fSpec = (type == 'd' ? &fwdRMDIR : &fwdRM);
       if (fSpec->Cmd)
          {if (Forward(retc, einfo, *fSpec, path, 0, info)) return retc;}
          else if ((retc = Finder->Locate(einfo,path,SFS_O_WRONLY|SFS_O_META)))
                  return fsError(einfo, retc);
      }

// Check if we should generate an event
//
   if (evsObject)
      {XrdOfsEvs::Event theEvent=(type == 'd' ? XrdOfsEvs::Rmdir:XrdOfsEvs::Rm);
       if (evsObject->Enabled(theEvent))
          {XrdOfsEvsInfo evInfo(tident, path, info, &rem_Env);
           evsObject->Notify(theEvent, evInfo);
          }
      }

// Perform the actual deletion
//
    retc = (type == 'd' ? XrdOfsOss->Remdir(path) : XrdOfsOss->Unlink(path));
    if (retc) return XrdOfsFS.Emsg(epname, einfo, retc, "remove", path);
    if (type == 'f')
       {XrdOfsHandle::Hide(path);
        if (Balancer) Balancer->Removed(path);
       }
    return SFS_OK;
}

/******************************************************************************/
/*                                r e n a m e                                 */
/******************************************************************************/

int XrdOfs::rename(const char             *old_name,  // In
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
   EPNAME("rename");
   int retc;
   const char *tident = einfo.getErrUser();
   XrdOucEnv old_Env(infoO);
   XrdOucEnv new_Env(infoN);
   XTRACE(rename, new_name, "old fn=" <<old_name <<" new ");

// Apply security, as needed
//
   AUTHORIZE2(client, einfo,
              AOP_Rename, "renaming",    old_name, &old_Env,
              AOP_Insert, "renaming to", new_name, &new_Env );

// Find out where we should rename this file
//
   if (Finder && Finder->isRemote())
      {if (fwdMV.Cmd)
          {if (Forward(retc, einfo, fwdMV, old_name, new_name, infoO, infoN))
              return retc;
          }
          else if ((retc = Finder->Locate(einfo,old_name,SFS_O_RDWR|SFS_O_META)))
                  return fsError(einfo, retc);
      }

// Check if we should generate an event
//
   if (evsObject && evsObject->Enabled(XrdOfsEvs::Mv))
      {XrdOfsEvsInfo evInfo(tident, old_name, infoO, &old_Env, 0, 0,
                                    new_name, infoN, &new_Env);
       evsObject->Notify(XrdOfsEvs::Mv, evInfo);
      }

// Perform actual rename operation
//
   if ((retc = XrdOfsOss->Rename(old_name, new_name)))
      return XrdOfsFS.Emsg(epname, einfo, retc, "rename", old_name);
   XrdOfsHandle::Hide(old_name);
   if (Balancer) {Balancer->Removed(old_name);
                  Balancer->Added(new_name);
                 }
   return SFS_OK;
}

/******************************************************************************/
/*                                  s t a t                                   */
/******************************************************************************/

int XrdOfs::stat(const char             *path,        // In
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
   EPNAME("stat");
   int retc;
   const char *tident = einfo.getErrUser();
   XrdOucEnv stat_Env(info);
   XTRACE(stat, path, "");

// Apply security, as needed
//
   AUTHORIZE(client,&stat_Env,AOP_Stat,"locate",path,einfo);

// Find out where we should stat this file
//
   if (Finder && Finder->isRemote()
   &&  (retc = Finder->Locate(einfo, path, SFS_O_RDONLY|SFS_O_STAT)))
      return fsError(einfo, retc);

// Now try to find the file or directory
//
   if ((retc = XrdOfsOss->Stat(path, buf)))
      return XrdOfsFS.Emsg(epname, einfo, retc, "locate", path);
   return SFS_OK;
}

/******************************************************************************/

int XrdOfs::stat(const char             *path,        // In
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
   EPNAME("stat");
   struct stat buf;
   int retc;
   const char *tident = einfo.getErrUser();
   XrdOucEnv stat_Env(info);
   XTRACE(stat, path, "");

// Apply security, as needed
//
   AUTHORIZE(client,&stat_Env,AOP_Stat,"locate",path,einfo);
   mode = (mode_t)-1;

// Find out where we should stat this file
//
   if (Finder && Finder->isRemote()
   &&  (retc = Finder->Locate(einfo,path,SFS_O_NOWAIT|SFS_O_RDONLY|SFS_O_STAT)))
      return fsError(einfo, retc);

// Now try to find the file or directory
//
   if (!(retc = XrdOfsOss->Stat(path, &buf, 1))) mode = buf.st_mode;
      else if ((-ENOMSG) != retc) return XrdOfsFS.Emsg(epname, einfo, retc,
                                                    "locate", path);
   return SFS_OK;
}

/******************************************************************************/
/*                              t r u n c a t e                               */
/******************************************************************************/

int XrdOfs::truncate(const char             *path,    // In
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
   EPNAME("truncate");
   const char *tident = einfo.getErrUser();
   XrdOucEnv trunc_Env(info);
   int retc;
   XTRACE(truncate, path, "");

// Apply security, as needed
//
   AUTHORIZE(client,&trunc_Env,AOP_Update,"truncate",path,einfo);

// Find out where we should chmod this file
//
   if (Finder && Finder->isRemote())
      {if (fwdTRUNC.Cmd)
          {char xSz[32];
           sprintf(xSz, "%lld", static_cast<long long>(Size));
           if (Forward(retc, einfo, fwdTRUNC, path, xSz, info)) return retc;
          }
          else if ((retc = Finder->Locate(einfo,path,SFS_O_RDWR)))
                  return fsError(einfo, retc);
      }

// Check if we should generate an event
//
   if (evsObject && evsObject->Enabled(XrdOfsEvs::Trunc))
      {XrdOfsEvsInfo evInfo(tident, path, info, &trunc_Env, 0, Size);
       evsObject->Notify(XrdOfsEvs::Trunc, evInfo);
      }

// Now try to find the file or directory
//
   if (!(retc = XrdOfsOss->Truncate(path, Size))) return SFS_OK;

// An error occured, return the error info
//
   return XrdOfsFS.Emsg(epname, einfo, retc, "trunc", path);
}

/******************************************************************************/
/*                                  E m s g                                   */
/******************************************************************************/

int XrdOfs::Emsg(const char    *pfx,    // Message prefix value
                 XrdOucErrInfo &einfo,  // Place to put text & error code
                 int            ecode,  // The error code
                 const char    *op,     // Operation being performed
                 const char    *target) // The target (e.g., fname)
{
   char *etext, buffer[XrdOucEI::Max_Error_Len], unkbuff[64];

// If the error is EBUSY then we just need to stall the client. This is
// a hack in order to provide for proxy support
//
    if (ecode < 0) ecode = -ecode;
    if (ecode == EBUSY) return 5;  // A hack for proxy support

// Check for timeout conditions that require a client delay
//
   if (ecode == ETIMEDOUT) return OSSDelay;

// Get the reason for the error
//
   if (!(etext = OfsEroute.ec2text(ecode))) 
      {sprintf(unkbuff, "reason unknown (%d)", ecode); etext = unkbuff;}

// Format the error message
//
    snprintf(buffer,sizeof(buffer),"Unable to %s %s; %s", op, target, etext);

// Print it out if debugging is enabled
//
#ifndef NODEBUG
    OfsEroute.Emsg(pfx, einfo.getErrUser(), buffer);
#endif

// Place the error message in the error object and return
//
    einfo.setErrInfo(ecode, buffer);
    return SFS_ERROR;
}

/******************************************************************************/
/*                     P R I V A T E    S E C T I O N                         */
/******************************************************************************/
/******************************************************************************/
/*                                 F n a m e                                  */
/******************************************************************************/

const char *XrdOfs::Fname(const char *path)
{
   int i = strlen(path)-1;
   while(i) if (path[i] == '/') return &path[i+1];
               else i--;
   return path;
}

/******************************************************************************/
/*                               F o r w a r d                                */
/******************************************************************************/
  
int XrdOfs::Forward(int &Result, XrdOucErrInfo &Resp, struct fwdOpt &Fwd,
                    const char *arg1, const char *arg2,
                    const char *arg3, const char *arg4)
{
   int retc;

   if ((retc = Finder->Forward(Resp, Fwd.Cmd, arg1, arg2, arg3, arg4)))
      {Result = fsError(Resp, retc);
       return 1;
      }

   if (Fwd.Port <= 0)
      {Result = SFS_OK;
       return (Fwd.Port ? 0 : 1);
      }

   Resp.setErrInfo(Fwd.Port, Fwd.Host);
   Result = SFS_REDIRECT;
   return 1;
}

/******************************************************************************/
/*                               f s E r r o r                                */
/******************************************************************************/
  
int XrdOfs::fsError(XrdOucErrInfo &myError, int rc)
{

// Translate the error code
//
   if (rc == -EREMOTE)     return SFS_REDIRECT;
   if (rc == -EINPROGRESS) return SFS_STARTED;
   if (rc > 0)             return rc;
   if (rc == -EALREADY)    return SFS_DATA;
                           return SFS_ERROR;
}

/******************************************************************************/
/*                                 S t a l l                                  */
/******************************************************************************/
  
int XrdOfs::Stall(XrdOucErrInfo   &einfo, // Error text & code
                  int              stime, // Seconds to stall
                  const char      *path)  // The path to stall on
{
    const char *msgfmt = "File %s is being staged; "
                         "estimated time to completion %s";
    EPNAME("Stall")
#ifndef NODEBUG
    const char *tident = "";
#endif
    char Mbuff[2048], Tbuff[32];

// Format the stall message
//
    snprintf(Mbuff, sizeof(Mbuff)-1, msgfmt,
             Fname(path), WaitTime(stime, Tbuff, sizeof(Tbuff)));
    ZTRACE(delay, "Stall " <<stime <<": " <<Mbuff <<" for " <<path);

// Place the error message in the error object and return
//
    einfo.setErrInfo(0, Mbuff);

// All done
//
   return (stime > MaxDelay ? MaxDelay : stime);
}
  
/******************************************************************************/
/*                              W a i t T i m e                               */
/******************************************************************************/

char *XrdOfs::WaitTime(int stime, char *buff, int blen)
{
   int mlen, hr, min, sec;

// Compute hours, minutes, and seconds
//
   min = stime / 60;
   sec = stime % 60;
   hr  = min   / 60;
   min = min   % 60;

// Now format the message based on time duration
//
        if (!hr && !min)
           mlen = snprintf(buff,blen,"%d second%s",sec,(sec > 1 ? "s" : ""));
   else if (!hr)
          {if (sec > 10) min++;
           mlen = snprintf(buff,blen,"%d minute%s",min,(min > 1 ? "s" : ""));
          }
   else   {if (hr == 1)
              if (min <= 30)
                      mlen = snprintf(buff,blen,"%d minutes",min+60);
                 else mlen = snprintf(buff,blen,"%d hour and %d minutes",hr,min);
              else {if (min > 30) hr++;
                      mlen = snprintf(buff,blen,"%d hours",hr);
                   }
          }

// Complete the message
//
   buff[blen-1] = '\0';
   return buff;
}
