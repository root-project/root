// @(#)root/rpdutils:$Name:  $:$Id: daemon.cxx,v 1.5 2002/10/28 14:22:51 rdm Exp $
// Author: Fons Rademakers   11/08/97
// Modifified: Gerardo Ganis 8/04/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// error                                                                //
//                                                                      //
// Set of error handling routines for daemon process.                   //
// Merging of rootd and proofd/src/error.cxx                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <syslog.h>
#include <errno.h>
#include <string.h>

#if defined(hpux9)
extern "C" {
   extern void openlog(const char *, int, int);
   extern void syslog(int, const char *, ...);
}
#endif

#ifdef __sun
#   ifndef _REENTRANT
#      if __SUNPRO_CC > 0x420
#         define GLOBAL_ERRNO
#      endif
#   endif
#endif

#include "rpderr.h"
#include "rpdp.h"

namespace ROOT {

extern int     gDebug;
extern int     gRootLog;

ErrorHandler_t gErrSys   = 0;
ErrorHandler_t gErrFatal = 0;
ErrorHandler_t gErr      = 0;


//______________________________________________________________________________
int GetErrno()
{
#ifdef GLOBAL_ERRNO
   return ::errno;
#else
   return errno;
#endif
}

//______________________________________________________________________________
void ResetErrno()
{
#ifdef GLOBAL_ERRNO
   ::errno = 0;
#else
   errno = 0;
#endif
}

//______________________________________________________________________________
void ErrorInfo(const char *va_(fmt), ...)
{
   // Write info message to syslog.

   char    buf[1024];
   va_list ap;

   va_start(ap,va_(fmt));
   vsprintf(buf, fmt, ap);
   va_end(ap);

   if (gRootLog == 0) {
    syslog(LOG_INFO, buf);
   } else if (gRootLog == 1) {
     fprintf(stderr, "%s\n",buf);
   } else if (gRootLog == 2) {
     if (strlen(gFileLog)>0) {
       FILE *fl= fopen(gFileLog,"a");
       fprintf(fl, "%s\n",buf);
       fclose(fl);
     }
   }
}

//______________________________________________________________________________
void ErrorInit(const char *ident)
{
   // Open syslog.

   openlog(ident, (LOG_PID | LOG_CONS), LOG_DAEMON);
}

//______________________________________________________________________________
void Perror(char *buf)
{
   // Return in buf the message belonging to errno.

   int len = strlen(buf);
#if (defined(__sun) && defined (__SVR4)) || defined (__linux) || \
   defined(_AIX) || defined(__MACH__)
   sprintf(buf+len, " (%s)", strerror(GetErrno()));
#else
   if (GetErrno() >= 0 && GetErrno() < sys_nerr)
      sprintf(buf+len, " (%s)", sys_errlist[GetErrno()]);
#endif
}

//______________________________________________________________________________
void Error(ErrorHandler_t func, int code, const char *va_(fmt), ...)
{
   // Write fatal message to syslog and exit.

   char    buf[1024];
   va_list ap;

   va_start(ap,va_(fmt));
   vsprintf(buf, fmt, ap);
   va_end(ap);

   if (gRootLog == 0) {
      syslog(LOG_ERR, buf);
   } else if (gRootLog == 1) {
      fprintf(stderr, "%s\n",buf);
   } else if (gRootLog == 2) {
      if (strlen(gFileLog)>0) {
         FILE *fl= fopen(gFileLog,"a");
         fprintf(fl, "%s\n",buf);
         fclose(fl);
      }
   }

   // Actions are defined by the specific error handler (see rootd.cxx and proofd.cxx)
   if (func) (*func)(code,(const char *)buf);
}

} // namespace ROOT
