// @(#)root/rootx:$Id$
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

#include "RConfigure.h"
#include "Rtypes.h"

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
#include <string>
#ifdef __APPLE__
#include <AvailabilityMacros.h>
#include <mach-o/dyld.h>
#endif

#if defined(__sgi) || defined(__sun)
#define HAVE_UTMPX_H
#define UTMP_NO_ADDR
#endif

#if defined(MAC_OS_X_VERSION_10_5)
#   define HAVE_UTMPX_H
#   define UTMP_NO_ADDR
#   ifndef ut_user
#      define ut_user ut_name
#   endif
#endif

#if (defined(__alpha) && !defined(__linux)) || defined(_AIX) || \
    defined(__FreeBSD__) || defined(__Lynx__) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && !defined(MAC_OS_X_VERSION_10_5))
#define UTMP_NO_ADDR
#endif

#ifdef __sun
#   ifndef _REENTRANT
#      if __SUNPRO_CC > 0x420
#         define GLOBAL_ERRNO
#      endif
#   endif
#endif

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

#if defined(__CYGWIN__) && defined(__GNUC__)
#define ROOTBINARY "root_exe.exe"
#else
#define ROOTBINARY "root.exe"
#endif

extern void PopupLogo(bool);
extern void WaitLogo();
extern void PopdownLogo();
extern void CloseDisplay();


static STRUCT_UTMP *gUtmpContents;
static bool gNoLogo = false;
static int childpid = -1;
const  int  kMAXPATHLEN = 8192;


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
   if (!gUtmpContents) {
      fclose(utmp);
      return 0;
   }

   n_read = fread(gUtmpContents, 1, size, utmp);
   if (!ferror(utmp)) {
      if (fclose(utmp) != EOF && n_read == size)
         return size / sizeof(STRUCT_UTMP);
   } else
      fclose(utmp);

   free(gUtmpContents);
   gUtmpContents = 0;
   return 0;
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

static const char *GetExePath()
{
   static std::string exepath;
   if (exepath == "") {
#ifdef __APPLE__
      exepath = _dyld_get_image_name(0);
#endif
#ifdef __linux
      char linkname[64];      // /proc/<pid>/exe
      char buf[kMAXPATHLEN];  // exe path name
      pid_t pid;

      // get our pid and build the name of the link in /proc
      pid = getpid();
      snprintf(linkname,64, "/proc/%i/exe", pid);
      int ret = readlink(linkname, buf, kMAXPATHLEN);
      if (ret > 0 && ret < kMAXPATHLEN) {
         buf[ret] = 0;
         exepath = buf;
      }
#endif
   }
   return exepath.c_str();
}

static void SetRootSys()
{
   const char *exepath = GetExePath();
   if (exepath && *exepath) {
      int l1 = strlen(exepath)+1;
      char *ep = new char[l1];
      strlcpy(ep, exepath, l1);
      char *s;
      if ((s = strrchr(ep, '/'))) {
         *s = 0;
         if ((s = strrchr(ep, '/'))) {
            *s = 0;
            int l2 = strlen(ep) + 10;
            char *env = new char[l2];
            snprintf(env, l2, "ROOTSYS=%s", ep);
            putenv(env);
         }
      }
      delete [] ep;
   }
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
            char *display = new char[sizeof(utmp_entry->ut_host) + 15];
            char *host = new char[sizeof(utmp_entry->ut_host) + 1];
            strncpy(host, utmp_entry->ut_host, sizeof(utmp_entry->ut_host));
            host[sizeof(utmp_entry->ut_host)] = 0;
            if (host[0]) {
               if (strchr(host, ':')) {
                  sprintf(display, "DISPLAY=%s", host);
                  fprintf(stderr, "*** DISPLAY not set, setting it to %s\n",
                          host);
               } else {
                  sprintf(display, "DISPLAY=%s:0.0", host);
                  fprintf(stderr, "*** DISPLAY not set, setting it to %s:0.0\n",
                          host);
               }
               putenv(display);
#ifndef UTMP_NO_ADDR
            } else if (utmp_entry->ut_addr) {
               struct hostent *he;
               if ((he = gethostbyaddr((const char*)&utmp_entry->ut_addr,
                                       sizeof(utmp_entry->ut_addr), AF_INET))) {
                  sprintf(display, "DISPLAY=%s:0.0", he->h_name);
                  fprintf(stderr, "*** DISPLAY not set, setting it to %s:0.0\n",
                          he->h_name);
                  putenv(display);
               }
#endif
            }
            delete [] host;
            // display cannot be deleted otherwise the env var is deleted too
         }
         free(gUtmpContents);
      }
   }
}

static void SetLibraryPath()
{
#ifndef ROOTLIBDIR
   // Set library path for the different platforms.

   char *msg;

#  if defined(__hpux)  || defined(_HIUX_SOURCE)
   if (getenv("SHLIB_PATH")) {
      msg = new char [strlen(getenv("ROOTSYS"))+strlen(getenv("SHLIB_PATH"))+100];
      sprintf(msg, "SHLIB_PATH=%s/lib:%s", getenv("ROOTSYS"),
                                           getenv("SHLIB_PATH"));
   } else {
      msg = new char [strlen(getenv("ROOTSYS"))+100];
      sprintf(msg, "SHLIB_PATH=%s/lib", getenv("ROOTSYS"));
   }
#  elif defined(_AIX)
   if (getenv("LIBPATH")) {
      msg = new char [strlen(getenv("ROOTSYS"))+strlen(getenv("LIBPATH"))+100];
      sprintf(msg, "LIBPATH=%s/lib:%s", getenv("ROOTSYS"),
                                        getenv("LIBPATH"));
   } else {
      msg = new char [strlen(getenv("ROOTSYS"))+100];
      sprintf(msg, "LIBPATH=%s/lib:/lib:/usr/lib", getenv("ROOTSYS"));
   }
#  elif defined(__APPLE__)
   if (getenv("DYLD_LIBRARY_PATH")) {
      msg = new char [strlen(getenv("ROOTSYS"))+strlen(getenv("DYLD_LIBRARY_PATH"))+100];
      sprintf(msg, "DYLD_LIBRARY_PATH=%s/lib:%s", getenv("ROOTSYS"),
                                                  getenv("DYLD_LIBRARY_PATH"));
   } else {
      msg = new char [strlen(getenv("ROOTSYS"))+100];
      sprintf(msg, "DYLD_LIBRARY_PATH=%s/lib", getenv("ROOTSYS"));
   }
#  else
   if (getenv("LD_LIBRARY_PATH")) {
      msg = new char [strlen(getenv("ROOTSYS"))+strlen(getenv("LD_LIBRARY_PATH"))+100];
      sprintf(msg, "LD_LIBRARY_PATH=%s/lib:%s", getenv("ROOTSYS"),
                                                getenv("LD_LIBRARY_PATH"));
   } else {
      msg = new char [strlen(getenv("ROOTSYS"))+100];
#  if defined(__sun)
      sprintf(msg, "LD_LIBRARY_PATH=%s/lib:/usr/dt/lib", getenv("ROOTSYS"));
#  else
      sprintf(msg, "LD_LIBRARY_PATH=%s/lib", getenv("ROOTSYS"));
#  endif
   }
#  endif
   putenv(msg);
#endif
}

extern "C" {
   static void SigUsr1(int);
   static void SigTerm(int);
}

static void SigUsr1(int)
{
   // When we get SIGUSR1 from child (i.e. ROOT) then pop down logo.

   if (!gNoLogo)
      PopdownLogo();
}

static void SigTerm(int sig)
{
   // When we get terminated, terminate child, too.
   kill(childpid, sig);
}

static void WaitChild()
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

   fprintf(stderr, "Usage: %s [-l] [-b] [-n] [-q] [dir] [[file:]data.root] [file1.C ... fileN.C]\n", pname);
   fprintf(stderr, "Options:\n");
   fprintf(stderr, "  -b : run in batch mode without graphics\n");
   fprintf(stderr, "  -n : do not execute logon and logoff macros as specified in .rootrc\n");
   fprintf(stderr, "  -q : exit after processing command line macro files\n");
   fprintf(stderr, "  -l : do not show splash screen\n");
   fprintf(stderr, "  -x : exit on exception\n");
   fprintf(stderr, " dir : if dir is a valid directory cd to it before executing\n");
   fprintf(stderr, "\n");
   fprintf(stderr, "  -?       : print usage\n");
   fprintf(stderr, "  -h       : print usage\n");
   fprintf(stderr, "  --help   : print usage\n");
   fprintf(stderr, "  -config  : print ./configure options\n");
   fprintf(stderr, "  -memstat : run with memory usage monitoring\n");
   fprintf(stderr, "\n");
}

int main(int argc, char **argv)
{
   char **argvv;
   char  arg0[kMAXPATHLEN];

#ifndef ROOTPREFIX
   // Try to set ROOTSYS depending on pathname of the executable
   SetRootSys();

   if (!getenv("ROOTSYS")) {
      fprintf(stderr, "%s: ROOTSYS not set. Set it before trying to run %s.\n",
              argv[0], argv[0]);
      return 1;
   }
#endif

   // In batch mode don't show splash screen, idem for no logo mode,
   // in about mode show always splash screen
   bool batch = false, about = false;
   int i;
   for (i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-?") || !strncmp(argv[i], "-h", 2) ||
          !strncmp(argv[i], "--help", 6)) {
         PrintUsage(argv[0]);
         return 1;
      }
      if (!strcmp(argv[i], "-b"))      batch   = true;
      if (!strcmp(argv[i], "-l"))      gNoLogo = true;
      if (!strcmp(argv[i], "-ll"))     gNoLogo = true;
      if (!strcmp(argv[i], "-a"))      about   = true;
      if (!strcmp(argv[i], "-config")) gNoLogo = true;
   }
   if (batch)
      gNoLogo = true;
   if (about) {
      batch   = false;
      gNoLogo = false;
   }

   if (!batch) {
      SetDisplay();
      if (!getenv("DISPLAY")) {
         fprintf(stderr, "%s: can't figure out DISPLAY, set it manually\n", argv[0]);
         fprintf(stderr, "In case you run a remote ssh session, restart your ssh session with:\n");
         fprintf(stderr, "=========>  ssh -Y\n");
         return 1;
      }
      if (about) {
         PopupLogo(true);
         WaitLogo();
         return 0;
      } else if (!gNoLogo)
         PopupLogo(false);
   }

   // Ignore SIGINT and SIGQUIT. Install handler for SIGUSR1.

   struct sigaction ignore, actUsr1, actTerm,
      saveintr, savequit, saveusr1, saveterm;

#if defined(__sun) && !defined(__i386) && !defined(__SVR4)
   ignore.sa_handler = (void (*)())SIG_IGN;
#elif defined(__sun) && defined(__SVR4)
   ignore.sa_handler = (void (*)(int))SIG_IGN;
#else
   ignore.sa_handler = SIG_IGN;
#endif
   sigemptyset(&ignore.sa_mask);
   ignore.sa_flags = 0;

   actUsr1 = ignore;
   actTerm = ignore;
#if defined(__sun) && !defined(__i386) && !defined(__SVR4)
   actUsr1.sa_handler = (void (*)())SigUsr1;
   actTerm.sa_handler = (void (*)())SigTerm;
#elif defined(__sun) && defined(__SVR4)
   actUsr1.sa_handler = SigUsr1;
   actTerm.sa_handler = SigTerm;
#elif (defined(__sgi) && !defined(__KCC)) || defined(__Lynx__)
#   if defined(IRIX64) || (__GNUG__>=3)
   actUsr1.sa_handler = SigUsr1;
   actTerm.sa_handler = SigTerm;
#   else
   actUsr1.sa_handler = (void (*)(...))SigUsr1;
   actTerm.sa_handler = (void (*)(...))SigTerm;
#   endif
#else
   actUsr1.sa_handler = SigUsr1;
   actTerm.sa_handler = SigTerm;
#endif
   sigaction(SIGINT,  &ignore, &saveintr);
   sigaction(SIGQUIT, &ignore, &savequit);
   sigaction(SIGUSR1, &actUsr1, &saveusr1);
   sigaction(SIGTERM, &actTerm, &saveterm);

   // Create child...

   if ((childpid = fork()) < 0) {
      fprintf(stderr, "%s: error forking child\n", argv[0]);
      return 1;
   } else if (childpid > 0) {
      if (!gNoLogo)
         WaitLogo();
      WaitChild();
   }

   // Continue with child...

   // Restore original signal actions
   sigaction(SIGINT,  &saveintr, 0);
   sigaction(SIGQUIT, &savequit, 0);
   sigaction(SIGUSR1, &saveusr1, 0);
   sigaction(SIGTERM, &saveterm, 0);

   // Close X display connection
   CloseDisplay();

   // Child is going to overlay itself with the actual ROOT module...

   // Build argv vector
   argvv = new char* [argc+2];
#ifdef ROOTBINDIR
   snprintf(arg0, sizeof(arg0), "%s/%s", ROOTBINDIR, ROOTBINARY);
#else
   snprintf(arg0, sizeof(arg0), "%s/bin/%s", getenv("ROOTSYS"), ROOTBINARY);
#endif
   argvv[0] = arg0;
   argvv[1] = (char *) "-splash";

   for (i = 1; i < argc; i++)
      argvv[1+i] = argv[i];
   argvv[1+i] = 0;

   // Make sure library path is set
   SetLibraryPath();

   // Execute actual ROOT module
   execv(arg0, argvv);

   // Exec failed
#ifndef ROOTBINDIR
   fprintf(stderr,
           "%s: can't start ROOT -- check that %s/bin/%s exists!\n",
           argv[0], getenv("ROOTSYS"), ROOTBINARY);
#else
   fprintf(stderr, "%s: can't start ROOT -- check that %s/%s exists!\n",
           argv[0], ROOTBINDIR, ROOTBINARY);
#endif

   return 1;
}
