/******************************************************************************/
/*                                                                            */
/*                   X r d X r o o t d P r e p a r e . c c                    */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdXrootdPrepareCVSID = "$Id$";
  
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/uio.h>

#ifdef __linux__
#include <syscall.h>
#define getdents(fd, dirp, cnt) syscall(SYS_getdents, fd, dirp, cnt)
#endif

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdXrootd/XrdXrootdPrepare.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

#ifndef NODEBUG  
extern XrdOucTrace     *XrdXrootdTrace;
#endif

       XrdScheduler    *XrdXrootdPrepare::SchedP;

       XrdSysError     *XrdXrootdPrepare::eDest;     // Error message handler

       int              XrdXrootdPrepare::scrubtime = 60*60;
       int              XrdXrootdPrepare::scrubkeep = 60*60*24;
       char            *XrdXrootdPrepare::LogDir = 0;
       int              XrdXrootdPrepare::LogDirLen = 0;
const  char            *XrdXrootdPrepare::TraceID = "Prepare";

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdXrootdPrepare::XrdXrootdPrepare(XrdSysError *errp, XrdScheduler *sp)
                 : XrdJob("Prep log scrubber")
{eDest    = errp;
 SchedP   = sp;
 if (LogDir) SchedP->Schedule((XrdJob *)this, scrubtime+time(0));
    else eDest->Say("Config warning: 'xrootd.prepare logdir' not specified; "
                    "prepare tracking disabled.");
}
  
/******************************************************************************/
/*                                  L i s t                                   */
/******************************************************************************/
  
int XrdXrootdPrepare::List(XrdXrootdPrepArgs &pargs, char *resp, int resplen)
{
   char *up, path[2048];
   struct dirent *dp;
   struct stat buf;
   int rc;

// If logging is not supported, return eof
//
   if (!LogDir) return -1;

// Check if this is the first call
//
   if (!pargs.dirP)
      {if (!(pargs.dirP = opendir((const char *)LogDir)))
          {eDest->Emsg("List", errno, "open prep log directory", LogDir);
           return -1;
          }
       if (pargs.reqid) pargs.reqlen = strlen(pargs.reqid);
       if (pargs.user)  pargs.usrlen = strlen(pargs.user);
      }

// Find the next entry that satisfies the search criteria
//
   errno = 0;
   while((dp = readdir(pargs.dirP)))
     {if (!(up = (char *) index((const char *)dp->d_name, '_'))) continue;
         if (pargs.reqlen && strncmp(dp->d_name, pargs.reqid, pargs.reqlen))
            continue;
         if (pargs.usrlen)
            if (!up || strcmp((const char *)up+1,(const char *)pargs.user))
               continue;
         strcpy(path, (const char *)LogDir);
         strcpy(path+LogDirLen, (const char *)dp->d_name);
         if (stat((const char *)path, &buf)) continue;
         *up = ' ';
         if ((up = (char *) index((const char *)(up+1), (int)'_'))) *up = ' ';
            else continue;
         if ((up = (char *) index((const char *)(up+1), (int)'_'))) *up = ' ';
            else continue;
         return snprintf(resp, resplen-1, "%s %ld", dp->d_name, buf.st_mtime);
        }

// Completed
//
   if ((rc = errno))
      eDest->Emsg("List", errno, "read prep log directory", LogDir);
   closedir(pargs.dirP);
   pargs.dirP = 0;
   return (rc ? -1 : 0);
}
  
/******************************************************************************/
/*                                   L o g                                    */
/******************************************************************************/
  
void XrdXrootdPrepare::Log(XrdXrootdPrepArgs &pargs)
{
   int rc, pnum = 0, xfd;
   XrdOucTList *tp = pargs.paths;
   char buff[2048], blink[2048];
   struct iovec iovec[2];

// If logging not enabled, return
//
   if (!LogDir) return;

// Count number of paths in the list
//
   while(tp) {pnum++; tp = tp->next;}

// Construct the file name: <reqid>_<user>_<prty>_<numpaths>
//
   snprintf(buff, sizeof(buff)-1, "%s%s_%s_%d_%d", LogDir,
                                  pargs.reqid, pargs.user, pargs.prty, pnum);

// Create the file
//
    if ((xfd = open(buff, O_WRONLY|O_CREAT|O_TRUNC,0644)) < 0)
       {eDest->Emsg("Log", errno, "open prep log file", buff);
        return;
       }

// Write all the paths into the file, separating each by a space
//
   iovec[1].iov_base = (char *)" ";
   iovec[1].iov_len  = 1;
   tp = pargs.paths;
   while(tp)
        {if (tp->next == 0) iovec[1].iov_base = (char *)"\n";
         iovec[0].iov_base = tp->text;
         iovec[0].iov_len  = strlen(tp->text);
         do {rc = writev(xfd, (const struct iovec *)iovec, 2);}
             while(rc < 0 && errno == EINTR);
         if (rc < 0)
            {eDest->Emsg("Log", errno, "write prep log file", buff);
             close(xfd);
             return;
            }
         tp = tp->next;
        }

// Create a symlink to the file
//
   close(xfd);
   strcpy(blink, LogDir); 
   strlcpy(blink+LogDirLen, pargs.reqid, sizeof(blink)-1);
   if (symlink((const char *)buff, (const char *)blink))
      {eDest->Emsg("Log", errno, "create symlink to prep log file", buff);
       return;
      }
}
  
/******************************************************************************/
/*                                L o g d e l                                 */
/******************************************************************************/
  
void XrdXrootdPrepare::Logdel(char *reqid)
{
   int rc;
   char path[MAXPATHLEN+256], buff[MAXPATHLEN+1];

// If logging not enabled, return
//
   if (!LogDir || strlen(reqid) > 255) return;

// Construct the file name of the symlink
//
   strcpy(path, (const char *)LogDir);
   strcpy(&path[LogDirLen], (const char *)reqid);

// Read the symlink contents for this request
//
   if ((rc = readlink((const char *)path, buff, sizeof(buff)-1)) < 0)
      {if (errno != ENOENT) eDest->Emsg("Logdel",errno,"read symlink",path);
       return;
      }

// Delete the file, then the symlink
//
   buff[rc] = '\0';
   if (unlink((const char *)buff)
   &&  errno != ENOENT) eDest->Emsg("Logdel",errno,"remove",buff);
      else TRACE(DEBUG, "Logdel removed " <<buff);
   if (unlink((const char *)path)
   &&  errno != ENOENT) eDest->Emsg("Logdel", errno, "remove", path);
      else TRACE(DEBUG, "Logdel removed " <<path);
}

/******************************************************************************/
/*                                  O p e n                                   */
/******************************************************************************/
  
int XrdXrootdPrepare::Open(const char *reqid, int &fsz)
{
   int fd;
   char path[MAXPATHLEN+264];
   struct stat buf;

// If logging is not supported, indicate so
//
   if (!LogDir) return -ENOTSUP;

// Construct the file name
//
   strcpy(path, (const char *)LogDir);
   strcpy(path+LogDirLen, reqid);

// Get the file size
//
   if (stat((const char *)path, &buf)) return -errno;
   fsz = buf.st_size;

// Open the file and return the file descriptor
//
   if ((fd = open((const char *)path, O_RDONLY)) < 0) return -errno;
   return fd;
}
  
/******************************************************************************/
/*                                 S c r u b                                  */
/******************************************************************************/
  
void XrdXrootdPrepare::Scrub()
{
   DIR *prepD;
   time_t stale = time(0) - scrubkeep;
   char *up, path[2048], *fn = path+LogDirLen;
   struct dirent *dp;
   struct stat buf;

// If logging is not supported, return eof
//
   if (!LogDir) return;

// Open the log directory
//
   if (!(prepD = opendir((const char *)LogDir)))
      {eDest->Emsg("Scrub", errno, "open prep log directory", LogDir);
       return;
      }
   strcpy(path, (const char *)LogDir);

// Delete all stale entries
//
   errno = 0;
   while((dp = readdir(prepD)))
     {if (!(up = (char *) index((const char *)dp->d_name, '_'))) continue;
         strcpy(fn, (const char *)dp->d_name);
         if (stat((const char *)path, &buf)) continue;
         if (buf.st_mtime <= stale)
            {TRACE(DEBUG, "Scrub removed stale prep log " <<path);
             unlink((const char *)path);
             *(fn+(up-dp->d_name)) = '\0';
             unlink((const char *)path);
             errno = 0;
            }
        }

// All done
//
   if (errno)
      eDest->Emsg("List", errno, "read prep log directory", LogDir);
   closedir(prepD);
}
 
/******************************************************************************/
/*                              s e t P a r m s                               */
/******************************************************************************/
  
int XrdXrootdPrepare::setParms(int stime, int keep)
{if (stime > 0) scrubtime = stime;
 if (keep  > 0) scrubkeep = keep;
 return 0;
}

int XrdXrootdPrepare::setParms(char *ldir)
{
   char path[2048];
   struct stat buf;
   int plen;

// If parm not supplied, ignore call
//
   if (!ldir) return 0;

// Make sure we have appropriate permissions for this directory
//
   if (access((const char *)ldir, X_OK | W_OK | R_OK) || stat(ldir, &buf))
      return -errno;
   if ((buf.st_mode & S_IFMT) != S_IFDIR) return -ENOTDIR;

// Create the path name
//
   if (LogDir) free(LogDir);
   LogDir = 0;
   plen = strlen(ldir);
   strcpy(path, ldir);
   if (path[plen-1] != '/') path[plen++] = '/';
   path[plen] = '\0';

// Save the path and return
//
   LogDir    = strdup(path);
   LogDirLen = strlen(LogDir);
   return 0;
}
