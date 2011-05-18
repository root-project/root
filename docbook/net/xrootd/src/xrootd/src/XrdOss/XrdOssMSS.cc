/******************************************************************************/
/*                                                                            */
/*                          X r d O s s M S S . c c                           */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/
  
//         $Id$

const char *XrdOssMSSCVSID = "$Id$";

#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <strings.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>
#if defined(AIX) || defined(__solaris__)
#include <sys/vnode.h>
#include <sys/mode.h>
#endif
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>

#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssTrace.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                  E r r o r   R o u t i n g   O b j e c t                   */
/******************************************************************************/
  
extern XrdSysError OssEroute;

extern XrdOucTrace OssTrace;

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define NegVal(x) (x <= 0 ? (x < 0 ? x : -1) : -x)

/******************************************************************************/
/*                           f i l e   h a n d l e                            */
/******************************************************************************/

/* These are private data structures. They are allocated dynamically to the
   appropriate size (yes, that means dbx has a tough time).
*/

struct XrdOssHandle
       {int hflag;
        XrdOucStream *sp;

        XrdOssHandle(int type, XrdOucStream *newsp=0) {hflag = type; sp = newsp;}
       ~XrdOssHandle() {if (sp) delete sp; hflag = 0;}

       };

#define XRDOSS_HT_EOF    1
#define XRDOSS_HT_DIR    4

/******************************************************************************/
/*          M a s s   S t o r a g e   R e l a t e d   M e t h o d s           */
/******************************************************************************/
  
/******************************************************************************/
/*                               o p e n d i r                                */
/******************************************************************************/
  
void *XrdOssSys::MSS_Opendir(const char *dir_path, int &rc) {
/*
  Function: Open the directory `path' and prepare for reading.

  Input:    path      - The fully qualified name of the directory to open.

  Output:   Returns a directory handle to be used for subsequent
            operations. If an error occurs, (-errno) is returned.
*/
     const char *epname = "MSS_Opendir";
     struct XrdOssHandle *oh;
     XrdOucStream *sp;

     // Make sure the path is not too long.
     //
     if (strlen(dir_path) > MAXPATHLEN)
        {OssEroute.Emsg(epname, "mss path too long - ", dir_path);
         rc = -ENAMETOOLONG;
         return (void *)0;
        }

     // Issue it now to trap any errors but defer reading the result until
     // readdir() is called. This does tie up a process, sigh.
     //
     if ( (rc = MSS_Xeq(&sp, ENOENT, "dlist", dir_path)))
        return (void *)0;

     // Allocate storage for the handle and return a copy of it.
     //
     if (!(oh = new XrdOssHandle(XRDOSS_HT_DIR, sp))) 
        {delete sp; rc = -ENOMEM; return (void *)0;}
     return (void *)oh;
}

/******************************************************************************/
/*                               r e a d d i r                                */
/******************************************************************************/

int XrdOssSys::MSS_Readdir(void *dir_handle, char *buff, int blen) {
/*
  Function: Read the next entry if directory 'dir_handle'.

  Input:    dir_handle - The value returned by a successful opendir() call.
            buff       - Buffer to hold directory name.
            blen       - Size of the buffer.

  Output:   Upon success, places the contents of the next directory entry
            in buff. When the end of the directory is encountered buff
            will be set to a null string.

            Upon failure, returns a (-errno).
*/
    const char *epname = "MSS_Readdir";
    int retc;
    struct XrdOssHandle *oh = (struct XrdOssHandle *)dir_handle;
    char *resp;

    // Verify that the handle is correct.
    //
    if ( !(oh->hflag & XRDOSS_HT_DIR) )
       {OssEroute.Emsg(epname, "invalid mss handle"); return -EBADF;}

    // Read a record from the directory, if possible.
    //
    if (oh->hflag & XRDOSS_HT_EOF) *buff = '\0';
       else if ((resp = oh->sp->GetLine()))
               {if ( ((int)strlen(resp)) >= blen )
                   {*buff = '\0';
                    return OssEroute.Emsg(epname, -EOVERFLOW,
                                            "readdir rmt", resp);
                   }
                   strlcpy(buff, resp, blen);
               } else {
                if ((retc = oh->sp->LastError())) return NegVal(retc);
                   else {*buff = '\0'; oh->hflag |= XRDOSS_HT_EOF;}
               }
    return XrdOssOK;
}

/******************************************************************************/
/*                              c l o s e d i r                               */
/******************************************************************************/
  
int XrdOssSys::MSS_Closedir(void *dir_handle) {
/*
  Function: Close the directory associated with handle "dir_handle".

  Input:    dir_handle - The handle returned by opendir().

  Output:   Returns 0 upon success and (-errno) upon failure.
*/
    const char *epname = "MSS_Closedir";
    struct XrdOssHandle *oh = (struct XrdOssHandle *)dir_handle;

    if ( !(oh->hflag & XRDOSS_HT_DIR) )
       {OssEroute.Emsg(epname, "invalid mss handle"); return -EBADF;}
    delete oh;
    return XrdOssOK;
}

/******************************************************************************/
/*                                c r e a t e                                 */
/******************************************************************************/

int XrdOssSys::MSS_Create(const char *path, mode_t file_mode, XrdOucEnv &env)
/*
  Function: Create a file named `path' with 'file_mode' access mode bits set.

  Input:    path      - The fully qualified name of the file to create.
            file_mode - The Posix access mode bits to be assigned to the file.
                        These bits correspond to the standard Unix permission
                        bits (e.g., 744 == "rwxr--r--").
            env         Enviornmental information.

  Output:   Returns zero upon success and (-errno) otherwise.
*/
{
    const char *epname = "MSS_Create";
    char myMode[16];

    // Make sure the path is not too long.
    //
    if (strlen(path) > MAXPATHLEN)
       {OssEroute.Emsg(epname, "mss path too long - ", path);
        return -ENAMETOOLONG;
       }

    // Construct the cmd to create the file. We currently don't support cosid.
    //
    sprintf(myMode, "%o", static_cast<int>(file_mode));

    // Create the file in in the mass store system
    //
    return MSS_Xeq(0, 0, "create", path, myMode);
}


/******************************************************************************/
/*                                 s t a t                                    */
/******************************************************************************/

/*
  Function: Determine if file 'path' actually exists.

  Input:    path        - Is the fully qualified name of the file to be tested.
            buff        - pointer to a 'stat' structure to hold the attributes
                          of the file.

  Output:   Returns 0 upon success and -errno upon failure.
*/

int XrdOssSys::MSS_Stat(const char *path, struct stat *buff)
{
    const char *epname = "MSS_Stat";
    char ftype, mtype[10], *resp;
    int retc, xt_nlink;
    long xt_uid, xt_gid, atime, ctime, mtime, xt_blksize, xt_blocks;
    long long xt_size;
    XrdOucStream *sfd;

    // Make sure the path is not too long.
    //
    if (strlen(path) > MAXPATHLEN)
       {OssEroute.Emsg(epname, "mss path too long - ", path);
        return -ENAMETOOLONG;
       }

    // Issue the command. This may be an immediate exists a or full statx.
    //
    if (!buff) return MSS_Xeq(0, ENOENT, (isMSSC ? "statx" : "exists"), path);
    if ((retc = MSS_Xeq(&sfd, ENOENT, "statx", path))) return retc;

    // Read in the results.
    //
    if ( !(resp = sfd ->GetLine()))
       return OssEroute.Emsg(epname,-XRDOSS_E8012,"process ",path);

    // Extract data from the response.
    //
    sscanf(resp, "%c %9s %d %ld %ld %ld %ld %ld %lld %ld %ld", &ftype, mtype,
           &xt_nlink, &xt_uid, &xt_gid, &atime, &ctime, &mtime,
           &xt_size, &xt_blksize, &xt_blocks);

    // Set the stat buffer, appropriately.
    //
    memset( (char *)buff, 0, sizeof(struct stat) );
    buff->st_nlink = static_cast<nlink_t>(xt_nlink);
    buff->st_uid   = static_cast<uid_t>(xt_uid);
    buff->st_gid   = static_cast<gid_t>(xt_gid);
    buff->st_atime = static_cast<time_t>(atime);
    buff->st_ctime = static_cast<time_t>(ctime);
    buff->st_mtime = static_cast<time_t>(mtime);
    buff->st_size  = static_cast<off_t>(xt_size);
    buff->st_blksize=static_cast<long>(xt_blksize);
#ifdef __macos__
    buff->st_blocks =                      xt_blocks;
#else
    buff->st_blocks =static_cast<blkcnt_t>(xt_blocks);
#endif

    if (ftype == 'd') buff->st_mode |=  S_IFDIR;
       else if (ftype == 'l') buff->st_mode |= S_IFLNK;
               else buff->st_mode |= S_IFREG;

    buff->st_mode |= tranmode(&mtype[0]) << 6;
    buff->st_mode |= tranmode(&mtype[3]) << 3;
    buff->st_mode |= tranmode(&mtype[6]);

    delete sfd;
    return 0;
}

int XrdOssSys::tranmode(char *mode) {
    int mbits = 0;
    if (mode[0] == 'r') mbits |= S_IROTH;
    if (mode[1] == 'w') mbits |= S_IWOTH;
    if (mode[2] == 'x') mbits |= S_IXOTH;
    return mbits;
}

/******************************************************************************/
/*                                r e m o v e                                 */
/******************************************************************************/

/*
  Function: Delete a file from the namespace and release it's data storage.

  Input:    path      - Is the fully qualified name of the file to be removed.

  Output:   Returns 0 upon success and -errno upon failure.
*/
int XrdOssSys::MSS_Unlink(const char *path) {
    const char *epname = "MSS_Unlink";

    // Make sure the path is not too long.
    //
    if (strlen(path) > MAXPATHLEN)
       {OssEroute.Emsg(epname, "mss path too long - ", path);
        return -ENAMETOOLONG;
       }

    // Remove the file in Mass Store System.
    //
    return MSS_Xeq(0, ENOENT, "rm", path);
}

/******************************************************************************/
/*                                r e n a m e                                 */
/******************************************************************************/

/*
  Function: Renames a file with name 'old_name' to 'new_name'.

  Input:    old_name  - Is the fully qualified name of the file to be renamed.
            new_name  - Is the fully qualified name that the file is to have.

  Output:   Returns 0 upon success and -errno upon failure.
*/
int XrdOssSys::MSS_Rename(const char *oldname, const char *newname) {
    const char *epname = "MSS_Rename";

    // Make sure the path is not too long.
    //
    if (strlen(oldname) > MAXPATHLEN
    ||  strlen(newname) > MAXPATHLEN)
       {OssEroute.Emsg(epname,"mss path too long - ", oldname, newname);
        return -ENAMETOOLONG;
       }

    // Rename the file in Mass Store System
    //
    return MSS_Xeq(0, 0, "mv", oldname, newname);
}

/******************************************************************************/
/*                     P R I V A T E    S E C T I O N                         */
/******************************************************************************/
/******************************************************************************/
/*                               M S S _ X e q                                */
/******************************************************************************/

int XrdOssSys::MSS_Xeq(XrdOucStream **xfd, int okerr, 
                       const char *cmd, const char *arg1, const char *arg2)
{
    EPNAME("MSS_Xeq")
    static int NoResp = 0;
    char *resp;
    int retc;
    XrdOucStream *sp;

// If we have no gateway command, return an error
//
   if (!RSSProg) return -XRDOSS_E8013;

// Allocate a stream for this command
//
   if (!(sp = new XrdOucStream(&OssEroute)))
      return OssEroute.Emsg("MSS_Xeq",-ENOMEM,"create stream for",RSSCmd);

// Run the command
//
   DEBUG("Invoking '" <<RSSCmd <<' ' <<cmd <<' ' <<(arg1 ? arg1 : "")
                      <<' ' <<(arg2 ? arg2 : ""));
   if ((retc = RSSProg->Run(sp, cmd, arg1, arg2)))
      {delete sp; return NegVal(retc);}

// Wait for data to appear. We do this to avoid hanging up and chewing through
// all of the threads while clients retry the requests with a new connection.
//
   if ((retc = sp->Wait4Data(RSSTout)))
      {if (retc < 0)
          {if (!(0xff & NoResp++))
               OssEroute.Emsg("MSS_Xeq", -ETIMEDOUT, "execute", cmd);
           retc = ETIMEDOUT;
          }
       delete sp; return NegVal(retc);
      }

// Read back the first record. The first records must be the return code
// from the command followed by any output. Make sure that this is the case.
//
   if ( !(resp = sp->GetLine()) ) retc = XRDOSS_E8023;
      else
      {DEBUG("received '" <<resp <<"'");
       if (sscanf(resp, "%d", &retc) <= 0) retc = XRDOSS_E8024;
      }
   if (retc)
      {if (retc != -okerr)
          OssEroute.Emsg("MSS_Xeq", NegVal(retc), "execute", cmd);
       delete sp;
       return NegVal(retc);
      }

// If the caller wants the stream pointer; return it. Otherwise close it.
//
   if (xfd) *xfd = sp;
      else delete sp;
   return 0;
}
