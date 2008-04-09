/******************************************************************************/
/*                                                                            */
/*                      X r d C S 2 D C M F i l e . c c                       */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/uio.h>
#include <utime.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>

#include "Xrd/XrdTrace.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdCS2/XrdCS2DCMFile.hh"

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

extern XrdSysError       XrdLog;

extern XrdOucTrace       XrdTrace;
 
/******************************************************************************/
/*                                C r e a t e                                 */
/******************************************************************************/
  
int XrdCS2DCMFile::Create(const char *thePath, char oMode, const char *Pfn)
{
   struct iovec iov[3] = {{&oMode,                          1}, // 0
                          {(void *)Pfn,           strlen(Pfn)}, // 1
                          {(char *)"\n",                    1}};// 2

   int fnfd;

// Create a file that will hold this information
//
   do {fnfd = open(thePath, O_RDWR|O_CREAT, S_IRUSR|S_IWUSR);}
      while( fnfd < 0 && errno == EINTR);
   if (fnfd < 0)
      {XrdLog.Emsg("Create", errno, "create file", thePath);
       return 0;
      }

// Write the information into the file
//
   if (writev(fnfd, iov, 3) < 0)
      {XrdLog.Emsg("Create", errno, "write file", thePath);
       close(fnfd);
       return 0;
      }

// All done here
//
   close(fnfd);
   return 1;
}

/******************************************************************************/
/*                                  I n i t                                   */
/******************************************************************************/

int XrdCS2DCMFile::Init(const char *thePath, time_t UpTime)
{
   const char *TraceID = "Init";
   struct stat buf;
   int rc, len;
  
// Read the contents of the file
//
   do {fd = open(thePath, O_RDONLY);} while(fd < 0 && errno == EINTR);
   if (fd < 0)
      {if ((rc = errno) != ENOENT)
          XrdLog.Emsg("Init", errno, "open file", thePath);
       return rc;
      }

   if (fstat(fd, &buf))
       {XrdLog.Emsg("Init", errno, "stat", thePath);
        close(fd); unlink(thePath);
        return 1;
       }

   if (UpTime && buf.st_ctime >= UpTime)
      {TRACE(DEBUG, "Skipping processing " <<thePath);
       close(fd);
       return EEXIST;
      } else {TRACE(DEBUG, "Processing " <<thePath);}

   if ((len = read(fd, fileData, sizeof(fileData)-1)) < 0)
      {rc = errno;
       XrdLog.Emsg("Init", errno, "read file", thePath);
       unlink(thePath);
       return rc;
      }

// Prepare to process the file
//
   close(fd); fd = -1;
   if (len < 2 || !(lp = index(fileData, '\n')))
      {XrdLog.Emsg("Init", "Invalid file data in", thePath);
       return EINVAL;
      }

   *lp = '\0'; lp++;
   isMod = (buf.st_mtime == 0)? 1:0 ; // Indicates file modified
   fileData[len] = '\0';
   myPath = thePath;
   return 0;
}

/******************************************************************************/
/*                                M o d i f y                                 */
/******************************************************************************/
  
void XrdCS2DCMFile::Modify(const char *thePath)
{
    struct utimbuf times;

// Indicate file modified by making access time and modtime zero
//
   XrdLog.Emsg("Modfied",0,"-< I am updating file", thePath);
   times.actime = times.modtime = 0;
   if (utime(thePath, (const struct utimbuf *)&times))
       XrdLog.Emsg("Modified", errno, "update time for file", thePath);
   XrdLog.Emsg("Modified",0,"-< I updated time for file", thePath);
}

/******************************************************************************/
/*                                 r e q I D                                  */
/******************************************************************************/
  
unsigned long long XrdCS2DCMFile::reqID()
{
   unsigned long long theID;

// Get the next request ID
//
   if (*lp)
      {theID = strtoull(lp, &np, 10);
       if (*np == '\n') {lp = np+1; return theID;}
       XrdLog.Emsg("reqID", "Invalid reqID", lp, myPath);
       lp = 0;
      }

   return 0;
}
