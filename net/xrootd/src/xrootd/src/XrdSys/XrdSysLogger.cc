/******************************************************************************/
/*                                                                            */
/*                       X r d S y s L o g g e r . c c                        */
/*                                                                            */
/*(c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University   */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*Produced by Andrew Hanushevsky for Stanford University under contract       */
/*           DE-AC03-76-SFO0515 with the Deprtment of Energy                  */
/******************************************************************************/

//       $Id$ 

const char *XrdSysLoggerCVSID = "$Id$";

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#ifndef WIN32
#include <dirent.h>
#include <unistd.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/termios.h>
#include <sys/uio.h>
#endif // WIN32

#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysTimer.hh"
 
/******************************************************************************/
/*                               S t a t i c s                                */
/******************************************************************************/
  
int XrdSysLogger::extLFD[4] = {-1, -1, -1, -1};

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdSysLogger::XrdSysLogger(int ErrFD, int dorotate)
{
   char * logFN;

   ePath = 0;
   eNTC  = 0;
   eInt  = 0;
   eNow  = 0;
   eFD   = ErrFD;
   eKeep = 0;
   doLFR = dorotate;

// Establish default log file name
//
   if (!(logFN = getenv("XrdSysLOGFILE"))) logFN = getenv("XrdOucLOGFILE");

// Establish message routing
//
   if (ErrFD != STDERR_FILENO) baseFD = ErrFD;
      else {baseFD = dup(ErrFD);
            fcntl(baseFD, F_SETFD, FD_CLOEXEC);
            Bind(logFN, 86400);
           }
}
  
/******************************************************************************/
/*                                  B i n d                                   */
/******************************************************************************/
  
int XrdSysLogger::Bind(const char *path, int isec)
{

// Compute time at midnight
//
   eNow = time(0);
   eNTC = XrdSysTimer::Midnight(eNow);

// Bind to the logfile as needed
//
   if (path) 
      {eInt  = isec;
       if (ePath) free(ePath);
       ePath = strdup(path);
       return ReBind(0);
      }
   eInt = 0;
   ePath = 0;
   return 0;
}

/******************************************************************************/
/*                                   P u t                                    */
/******************************************************************************/
  
void XrdSysLogger::Put(int iovcnt, struct iovec *iov)
{
    int retc;
    char tbuff[24];

// Prefix message with time if calle wants it so
//
   if (iov[0].iov_base) eNow = time(0);
      else {iov[0].iov_base = tbuff;
            iov[0].iov_len  = (int)Time(tbuff);
           }

// Obtain the serailization mutex if need be
//
   Logger_Mutex.Lock();

// Check if we should close and reopen the output
//
   if (eInt && eNow >= eNTC) ReBind();

// In theory, writev may write out a partial list. This rarely happens in
// practice and so we ignore that possibility (recovery is pretty tough).
//
   do { retc = writev(eFD, (const struct iovec *)iov, iovcnt);}
               while (retc < 0 && errno == EINTR);

// Release the serailization mutex if need be
//
   Logger_Mutex.UnLock();
}

/******************************************************************************/
/*                                  T i m e                                   */
/******************************************************************************/
  
int XrdSysLogger::Time(char *tbuff)
{
    const int minblen = 24;
    eNow = time(0);
    struct tm tNow;
    int i;

// Format the header
//
   tbuff[minblen-1] = '\0'; // tbuff must be at least 24 bytes long
   localtime_r((const time_t *) &eNow, &tNow);
   i =    snprintf(tbuff, minblen, "%02d%02d%02d %02d:%02d:%02d %03ld ",
                  tNow.tm_year-100, tNow.tm_mon+1, tNow.tm_mday,
                  tNow.tm_hour,     tNow.tm_min,   tNow.tm_sec,
                  XrdSysThread::Num());
   return (i >= minblen ? minblen-1 : i);
}

/******************************************************************************/
/*                                x l o g F D                                 */
/******************************************************************************/
  
int XrdSysLogger::xlogFD() {return -1;}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               p u t E m s g                                */
/******************************************************************************/
  
// This internal logging method is used when the caller already has the mutex!

void XrdSysLogger::putEmsg(char *msg, int msz)
{
    struct iovec eVec[2];
    int retc;
    char tbuff[24];

// Prefix message with time
//
   eVec[0].iov_base = tbuff;
   eVec[0].iov_len  = (int)Time(tbuff);
   eVec[1].iov_base = msg;
   eVec[1].iov_len  = msz;

// In theory, writev may write out a partial list. This rarely happens in
// practice and so we ignore that possibility (recovery is pretty tough).
//
   do { retc = writev(eFD, (const struct iovec *)eVec, 2);}
               while (retc < 0 && errno == EINTR);
}

/******************************************************************************/
/*                                R e B i n d                                 */
/******************************************************************************/
  
int XrdSysLogger::ReBind(int dorename)
{
   const char seq[] = "0123456789";
   unsigned int i;
   int newfd;
   struct tm nowtime;
   char *bp, buff[MAXPATHLEN+MAXNAMELEN];
   struct stat bf;

// Rename the file to be of the form yyyymmdd corresponding to the date it was
// opened. We will add a sequence number (.x) if a conflict occurs.
//
   if (dorename && doLFR)
      {strcpy(buff, ePath);
       bp = buff+strlen(ePath);
       *bp++ = '.';
       strncpy(bp, Filesfx, 8);
       bp += 8;
       *bp = '\0'; *(bp+2) = '\0';
       for (i = 0; i < sizeof(seq) && !stat(buff, &bf); i++)
           {*bp = '.'; *(bp+1) = (char)seq[i];}
       if (i < sizeof(seq)) rename(ePath, buff);
      }

// Compute the new suffix
//
   localtime_r((const time_t *) &eNow, &nowtime);
   sprintf(buff, "%4d%02d%02d", nowtime.tm_year+1900, nowtime.tm_mon+1,
                                nowtime.tm_mday);
   strncpy(Filesfx, buff, 8);

// Set new close interval
//
   if (eInt > 0) while(eNTC <= eNow) eNTC += eInt;

// Open the file for output. Note that we can still leak a file descriptor
// if a thread forks a process before we are able to do the fcntl(), sigh.
//
   if ((newfd = open(ePath,O_WRONLY|O_APPEND|O_CREAT,0644)) < 0) return -errno;
   fcntl(newfd, F_SETFD, FD_CLOEXEC);

// Now set the file descriptor to be the same as the error FD. This will
// close the previously opened file, if any.
//
   if (dup2(newfd, eFD) < 0) return -errno;
   close(newfd);

// Check if we should trim log files
//
   if (eKeep && doLFR) Trim();
   return 0;
}

/******************************************************************************/
/*                                  T r i m                                   */
/******************************************************************************/

#ifndef WIN32
void XrdSysLogger::Trim()
{
   struct LogFile 
          {LogFile *next;
           char    *fn;
           off_t    sz;
           time_t   tm;

           LogFile(char *xfn, off_t xsz, time_t xtm)
                  {fn = (xfn ? strdup(xfn) : 0); sz = xsz; tm = xtm; next = 0;}
          ~LogFile() 
                  {if (fn)   free(fn);
                   if (next) delete next;
                  }
          } logList(0,0,0);

   struct LogFile *logEnt, *logPrev, *logNow;
   char eBuff[2048], logFN[MAXNAMELEN+8], logDir[MAXPATHLEN+8], *logSfx;
   struct dirent *dp;
   struct stat buff;
   long long totSz = 0;
   int n,rc, totNum= 0;
   DIR *DFD;

// Ignore this call if we are not deleting log files
//
   if (!eKeep) return;

// Construct the directory path
//
   if (!ePath) return;
   strcpy(logDir, ePath);
   if (!(logSfx = rindex(logDir, '/'))) return;
   *logSfx = '\0';
   strcpy(logFN, logSfx+1);
   n = strlen(logFN);

// Open the directory
//
   if (!(DFD = opendir(logDir)))
      {int msz = sprintf(eBuff, "Error %d (%s) opening log directory %s\n",
                                errno, strerror(errno), logDir);
       putEmsg(eBuff, msz);
       return;
      }
    *logSfx++ = '/';

// Record all of the log files currently in this directory
//
   errno = 0;
   while((dp = readdir(DFD)))
        {if (strncmp(dp->d_name, logFN, n)) continue;
         strcpy(logSfx, dp->d_name);
         if (stat(logDir, &buff) || !(buff.st_mode & S_IFREG)) continue;

         totNum++; totSz += buff.st_size;
         logEnt = new LogFile(dp->d_name, buff.st_size, buff.st_mtime);
         logPrev = &logList; logNow = logList.next;
         while(logNow && logNow->tm < buff.st_mtime)
              {logPrev = logNow; logNow = logNow->next;}

         logPrev->next = logEnt; 
         logEnt->next  = logNow;
        }

// Check if we received an error
//
   rc = errno; closedir(DFD);
   if (rc)
      {int msz = sprintf(eBuff, "Error %d (%s) reading log directory %s\n",
                                rc, strerror(rc), logDir);
       putEmsg(eBuff, msz);
       return;
      }

// If there is only one log file here no need to
//
   if (totNum <= 1) return;

// Check if we need to trim log files
//
   if (eKeep < 0)
      {if ((totNum += eKeep) <= 0) return;
      } else {
       if (totSz <= eKeep)         return;
       logNow = logList.next; totNum = 0;
       while(logNow && totSz > eKeep)
            {totNum++; totSz -= logNow->sz; logNow = logNow->next;}
      }

// Now start deleting log files
//
   logNow = logList.next;
   while(logNow && totNum--)
        {strcpy(logSfx, logNow->fn);
         if (unlink(logDir))
            rc = sprintf(eBuff, "Error %d (%s) removing log file %s\n",
                                errno, strerror(errno), logDir);
            else rc = sprintf(eBuff, "Removed log file %s\n", logDir);
         putEmsg(eBuff, rc);
         logNow = logNow->next;
        }
}
#else
void XrdSysLogger::Trim()
{
}
#endif
