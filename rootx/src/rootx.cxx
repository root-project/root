// @(#)root/rootx:$Name:  $:$Id: rootx.cxx,v 1.8 2001/11/06 13:38:27 rdm Exp $
// Author: Fons Rademakers   19/02/98

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rootx                                                                //
//                                                                      //
// Rootx is a small front-end program that starts the main ROOT module. //
// This program is called "root" in the $ROOTSYS/bin directory and the  //
// real ROOT executable is now called "root.exe" (formerly "root").     //
// Rootx puts up a splash screen giving some info about the current     //
// version of ROOT and, more importanly, it sets up the required        //
// LD_LIBRARY_PATH, SHLIB_PATH and LIBPATH environment variables        //
// (depending on the platform).                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <errno.h>

#include <netdb.h>
#include <sys/socket.h>

#if defined(__sgi) || defined(__sun)
#define HAVE_UTMPX_H
#define UTMP_NO_ADDR
#endif
#if (defined(__alpha) && !defined(__linux)) || defined(_AIX) || \
    defined(__FreeBSD__) || defined(__Lynx__) || defined(__APPLE__)
#define UTMP_NO_ADDR
#endif

#ifdef __sun
#   ifndef _REENTRANT
#      if __SUNPRO_CC > 0x420
#         define GLOBAL_ERRNO
#      endif
#   endif
#endif

#ifndef __VMS
# ifdef HAVE_UTMPX_H
# include <utmpx.h>
# define STRUCT_UTMP struct utmpx
# else
# if defined(__linux) && defined(__powerpc) && (__GNUC__ == 2) && (__GNUC_MINOR__ < 90)
   extern "C" {
# endif
# include <utmp.h>
# define STRUCT_UTMP struct utmp
# endif
#endif

#if !defined(UTMP_FILE) && defined(_PATH_UTMP)      // 4.4BSD
#define UTMP_FILE _PATH_UTMP
#endif
#if defined(UTMPX_FILE)                             // Solaris, SysVr4
#undef  UTMP_FILE
#define UTMP_FILE UTMPX_FILE
#endif
#ifndef UTMP_FILE
#define UTMP_FILE "/etc/utmp"
#endif

extern void PopupLogo();
extern void PopdownLogo();
extern void CloseDisplay();


static STRUCT_UTMP *gUtmpContents;
static int          gBatch  = 0;
static int          gNoLogo = 0;


static int GetErrno()
{
#ifdef GLOBAL_ERRNO
   return ::errno;
#else
   return errno;
#endif
}

static void ResetErrno()
{
#ifdef GLOBAL_ERRNO
   ::errno = 0;
#else
   errno = 0;
#endif
}

static int ReadUtmp()
{
   FILE  *utmp;
   struct stat file_stats;
   size_t n_read, size;

   gUtmpContents = 0;

   utmp = fopen(UTMP_FILE, "r");
   if (!utmp)
      return 0;

   fstat(fileno(utmp), &file_stats);
   size = file_stats.st_size;
   if (size <= 0) {
      fclose(utmp);
      return 0;
   }

   gUtmpContents = (STRUCT_UTMP *) malloc(size);
   if (!gUtmpContents) return 0;

   n_read = fread(gUtmpContents, 1, size, utmp);
   if (ferror(utmp) || fclose(utmp) == EOF || n_read < size) {
      free(gUtmpContents);
      gUtmpContents = 0;
      return 0;
   }

   return size / sizeof(STRUCT_UTMP);
}

static STRUCT_UTMP *SearchEntry(int n, const char *tty)
{
   STRUCT_UTMP *ue = gUtmpContents;

   while (n--) {
      if (ue->ut_name[0] && !strncmp(tty, ue->ut_line, sizeof(ue->ut_line)))
         return ue;
      ue++;
   }
   return 0;
}

static void SetDisplay()
{
   // Set DISPLAY environment variable.

   if (!getenv("DISPLAY")) {
      char *tty = ttyname(0);  // device user is logged in on
      if (tty) {
         tty += 5;             // remove "/dev/"
         STRUCT_UTMP *utmp_entry = SearchEntry(ReadUtmp(), tty);
         if (utmp_entry) {
            static char display[64];
            if (utmp_entry->ut_host[0]) {
               if (strchr(utmp_entry->ut_host, ':')) {
                  sprintf(display, "DISPLAY=%s", utmp_entry->ut_host);
                  fprintf(stderr, "*** DISPLAY not set, setting it to %s\n",
                          utmp_entry->ut_host);
	       } else {
                  sprintf(display, "DISPLAY=%s:0.0", utmp_entry->ut_host);
                  fprintf(stderr, "*** DISPLAY not set, setting it to %s:0.0\n",
                          utmp_entry->ut_host);
               }
               putenv((char *)display);
#ifndef UTMP_NO_ADDR
            } else if (utmp_entry->ut_addr) {
               struct hostent *he;
               if ((he = gethostbyaddr((const char*)&utmp_entry->ut_addr,
                                       sizeof(utmp_entry->ut_addr), AF_INET))) {
                  sprintf(display, "DISPLAY=%s:0.0", he->h_name);
                  fprintf(stderr, "*** DISPLAY not set, setting it to %s:0.0\n",
                          he->h_name);
                  putenv((char *)display);
               }
#endif
            }
         }
         free(gUtmpContents);
      }
   }
}

static void SetLibraryPath()
{
#ifndef ROOTLIBDIR
   // Set library path for the different platforms.

   static char msg[512];

#  if defined(__linux) || defined(__alpha) || defined(__sgi) || \
      defined(__sun) || defined(__FreeBSD__)
   if (getenv("LD_LIBRARY_PATH"))
      sprintf(msg, "LD_LIBRARY_PATH=%s/lib:%s", getenv("ROOTSYS"),
                                                getenv("LD_LIBRARY_PATH"));
   else
#  if defined(__sun)
      sprintf(msg, "LD_LIBRARY_PATH=%s/lib:/usr/dt/lib", getenv("ROOTSYS"));
#  else
      sprintf(msg, "LD_LIBRARY_PATH=%s/lib", getenv("ROOTSYS"));
#  endif
#  elif defined(__hpux)  || defined(_HIUX_SOURCE)
   if (getenv("SHLIB_PATH"))
      sprintf(msg, "SHLIB_PATH=%s/lib:%s", getenv("ROOTSYS"),
                                           getenv("SHLIB_PATH"));
   else
      sprintf(msg, "SHLIB_PATH=%s/lib", getenv("ROOTSYS"));
#  elif defined(_AIX)
   if (getenv("LIBPATH"))
      sprintf(msg, "LIBPATH=%s/lib:%s", getenv("ROOTSYS"),
                                        getenv("LIBPATH"));
   else
      sprintf(msg, "LIBPATH=%s/lib:/lib:/usr/lib", getenv("ROOTSYS"));
#  endif
   putenv((char *)msg);
#endif
}

extern "C" {
   static void SigUsr1(int);
}

static void SigUsr1(int)
{
   // When we get SIGUSR1 from child (i.e. ROOT) then pop down logo.

   if (!gBatch)
      PopdownLogo();
}

static void WaitChild(int childpid)
{
   // Wait till child (i.e. ROOT) is finished.

   int status;

   do {
      while (waitpid(childpid, &status, WUNTRACED) < 0) {
         if (GetErrno() != EINTR)
            break;
         ResetErrno();
      }

      if (WIFEXITED(status))
         exit(WEXITSTATUS(status));

      if (WIFSIGNALED(status))
         exit(WTERMSIG(status));

      if (WIFSTOPPED(status)) {         // child got ctlr-Z
         raise(SIGTSTP);                // stop also parent
         kill(childpid, SIGCONT);       // if parent wakes up, wake up child
      }
   } while (WIFSTOPPED(status));

   exit(0);
}

static void PrintUsage(char *pname)
{
   // This is a copy of the text in TApplication::GetOptions().

   fprintf(stderr, "Usage: %s [-l] [-b] [-n] [-q] [dir] [file1.C ... fileN.C]\n", pname);
   fprintf(stderr, "Options:\n");
   fprintf(stderr, "  -b : run in batch mode without graphics\n");
   fprintf(stderr, "  -n : do not execute logon and logoff macros as specified in .rootrc\n");
   fprintf(stderr, "  -q : exit after processing command line macro files\n");
   fprintf(stderr, "  -l : do not show splash screen\n");
   fprintf(stderr, " dir : if dir is a valid directory cd to it before executing\n");
   fprintf(stderr, "\n");
}

int main(int argc, char **argv)
{
   const int kMAXARGS = 256;
   char *argvv[kMAXARGS];
   char  arg0[256];

#ifndef ROOTPREFIX
   if (!getenv("ROOTSYS")) {
      fprintf(stderr, "%s: ROOTSYS not set. Set it before trying to run %s.\n",
              argv[0], argv[0]);
      return 1;
   }
#endif

   // In batch mode don't show splash screen
   int i;
   for (i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-?") || !strncmp(argv[i], "-h", 2)) {
         PrintUsage(argv[0]);
         return 1;
      }
      if (!strcmp(argv[i], "-b")) gBatch  = 1;
      if (!strcmp(argv[i], "-l")) gNoLogo = 1;
   }

   if (!gBatch) {
      SetDisplay();
      if (!getenv("DISPLAY")) {
         fprintf(stderr, "%s: can't figure out DISPLAY, set it manually\n", argv[0]);
         return 1;
      }
      if (!gNoLogo) PopupLogo();
   }

   // Ignore SIGINT and SIGQUIT. Install handler for SIGUSR1.

   struct sigaction ignore, handle, saveintr, savequit, saveusr1;

#if defined(__sun) && !defined(__i386) && !defined(__SVR4)
   ignore.sa_handler = (void (*)())SIG_IGN;
#elif defined(__sun) && defined(__SVR4)
   ignore.sa_handler = (void (*)(int))SIG_IGN;
#else
   ignore.sa_handler = SIG_IGN;
#endif
   sigemptyset(&ignore.sa_mask);
   ignore.sa_flags = 0;
   handle = ignore;
#if defined(__sun) && !defined(__i386) && !defined(__SVR4)
   handle.sa_handler = (void (*)())SigUsr1;
#elif defined(__sun) && defined(__SVR4)
   handle.sa_handler = SigUsr1;
#elif (defined(__sgi) && !defined(__KCC)) || defined(__Lynx__)
#   if defined(IRIX64)
   handle.sa_handler = SigUsr1;
#   else
   handle.sa_handler = (void (*)(...))SigUsr1;
#   endif
#else
   handle.sa_handler = SigUsr1;
#endif
   sigaction(SIGINT,  &ignore, &saveintr);
   sigaction(SIGQUIT, &ignore, &savequit);
   sigaction(SIGUSR1, &handle, &saveusr1);

   // Create child...

   int childpid;
   if ((childpid = fork()) < 0) {
      fprintf(stderr, "%s: error forking child\n", argv[0]);
      return 1;
   } else if (childpid > 0)
      WaitChild(childpid);

   // Continue with child...

   // Restore original signal actions
   sigaction(SIGINT,  &saveintr, 0);
   sigaction(SIGQUIT, &savequit, 0);
   sigaction(SIGUSR1, &saveusr1, 0);

   // Close X display connection
   CloseDisplay();

   // Child is going to overlay itself with the actual ROOT module...

   // Build argv vector
#ifdef ROOTBINDIR
   sprintf(arg0, "%s/root.exe", ROOTBINDIR);
#else
   sprintf(arg0, "%s/bin/root.exe", getenv("ROOTSYS"));
#endif
   argvv[0] = arg0;
   argvv[1] = (char *) "-splash";

   int iargc = argc;
   if (iargc > kMAXARGS-2) iargc = kMAXARGS-2;
   for (i = 1; i < iargc; i++)
      argvv[1+i] = argv[i];
   argvv[1+i] = 0;

   // Make sure library path is set
   SetLibraryPath();

   // Execute actual ROOT module
   execv(arg0, argvv);

   // Exec failed
#ifndef ROOTBINDIR
   fprintf(stderr,
	   "%s: can't start ROOT -- check that %s/bin/root.exe exists!\n",
           argv[0], getenv("ROOTSYS"));
#else
   fprintf(stderr, "%s: can't start ROOT -- check that %s/root.exe exists!\n",
           argv[0], ROOTBINDIR);
#endif

   return 1;
}

