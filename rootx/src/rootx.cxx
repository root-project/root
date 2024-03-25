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
// version of ROOT and, more importantly, it sets up the required       //
// LD_LIBRARY_PATH, SHLIB_PATH and LIBPATH environment variables        //
// (depending on the platform).                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"
#include "Rtypes.h"
#include "snprintf.h"
#include "rootCommandLineOptionsHelp.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <signal.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <cerrno>
#include <netdb.h>
#include <sys/socket.h>
#include <string>
#ifdef __APPLE__
#include <AvailabilityMacros.h>
#include <mach-o/dyld.h>
#endif

#ifdef R__FBSD
#include <sys/param.h>
#include <sys/user.h>
#include <sys/types.h>
#include <libutil.h>
#include <libprocstat.h>
#endif // R__FBSD

#if defined(__CYGWIN__) && defined(__GNUC__)
#define ROOTBINARY "root_exe.exe"
#else
#define ROOTBINARY "root.exe"
#endif

static int gChildpid;
static int GetErrno()
{
   return errno;
}

static void ResetErrno()
{
   errno = 0;
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
#if defined(R__FBSD)
  procstat* ps = procstat_open_sysctl();  //
  kinfo_proc* kp = kinfo_getproc(getpid());

  if (kp!=NULL) {
     char path_str[PATH_MAX] = "";
     procstat_getpathname(ps, kp, path_str, sizeof(path_str));
     exepath = path_str;
  }

  free(kp);
  procstat_close(ps);
#endif
   }
   return exepath.c_str();
}

static void SetRootSys()
{
   const char *exepath = GetExePath();
   if (exepath && *exepath) {
      std::string epStr{exepath};
      char *ep = epStr.data();
      char *s;
      if ((s = strrchr(ep, '/'))) {
         *s = 0;
         if ((s = strrchr(ep, '/'))) {
            *s = 0;
            int l2 = strlen(ep) + 10;
            char *env = new char[l2];
            snprintf(env, l2, "ROOTSYS=%s", ep);
            putenv(env); // NOLINT: allocated memory now used by environment variable
         }
      }
   }
}

#ifndef IS_RPATH_BUILD
static void SetLibraryPath()
{
# ifdef ROOTPREFIX
   if (getenv("ROOTIGNOREPREFIX")) {
# endif
   // Set library path for the different platforms.

      char *msg;

#if defined(__hpux) || defined(_HIUX_SOURCE)
      if (getenv("SHLIB_PATH")) {
         const auto msgLen = strlen(getenv("ROOTSYS")) + strlen(getenv("SHLIB_PATH")) + 100;
         msg = new char[msgLen];
         snprintf(msg, msgLen, "SHLIB_PATH=%s/lib:%s", getenv("ROOTSYS"), getenv("SHLIB_PATH"));
      } else {
         const auto msgLen = strlen(getenv("ROOTSYS")) + 100;
         msg = new char[msgLen];
         snprintf(msg, msgLen, "SHLIB_PATH=%s/lib", getenv("ROOTSYS"));
      }
#elif defined(_AIX)
   if (getenv("LIBPATH")) {
      const auto msgLen = strlen(getenv("ROOTSYS")) + strlen(getenv("LIBPATH")) + 100;
      msg = new char[msgLen];
      snprintf(msg, msgLen, "LIBPATH=%s/lib:%s", getenv("ROOTSYS"), getenv("LIBPATH"));
   } else {
      const auto msgLen = strlen(getenv("ROOTSYS")) + 100;
      msg = new char[msgLen];
      snprintf(msg, msgLen, "LIBPATH=%s/lib:/lib:/usr/lib", getenv("ROOTSYS"));
   }
#elif defined(__APPLE__)
   if (getenv("DYLD_LIBRARY_PATH")) {
      const auto msgLen = strlen(getenv("ROOTSYS")) + strlen(getenv("DYLD_LIBRARY_PATH")) + 100;
      msg = new char[msgLen];
      snprintf(msg, msgLen, "DYLD_LIBRARY_PATH=%s/lib:%s", getenv("ROOTSYS"), getenv("DYLD_LIBRARY_PATH"));
   } else {
      const auto msgLen = strlen(getenv("ROOTSYS")) + 100;
      msg = new char[msgLen];
      snprintf(msg, msgLen, "DYLD_LIBRARY_PATH=%s/lib", getenv("ROOTSYS"));
   }
#else
   if (getenv("LD_LIBRARY_PATH")) {
      const auto msgLen = strlen(getenv("ROOTSYS")) + strlen(getenv("LD_LIBRARY_PATH")) + 100;
      msg = new char[msgLen];
      snprintf(msg, msgLen, "LD_LIBRARY_PATH=%s/lib:%s", getenv("ROOTSYS"), getenv("LD_LIBRARY_PATH"));
   } else {
      const auto msgLen = strlen(getenv("ROOTSYS")) + 100;
      msg = new char[msgLen];
#if defined(__sun)
      snprintf(msg, msgLen, "LD_LIBRARY_PATH=%s/lib:/usr/dt/lib", getenv("ROOTSYS"));
#else
      snprintf(msg, msgLen, "LD_LIBRARY_PATH=%s/lib", getenv("ROOTSYS"));
#endif
   }
#endif
      putenv(msg);
# ifdef ROOTPREFIX
   } else /* if (getenv("ROOTIGNOREPREFIX")) */ {
      std::string ldLibPath = "LD_LIBRARY_PATH=" ROOTLIBDIR;
      if (const char *oldLdLibPath = getenv("LD_LIBRARY_PATH"))
         ldLibPath += std::string(":") + oldLdLibPath;
      char *msg = strdup(ldLibPath.c_str());
      putenv(msg);
   }
# endif
}
#endif

extern "C" {
   static void SigTerm(int);
}

static void SigTerm(int sig)
{
   // When we get terminated, terminate child, too.
   kill(gChildpid, sig);
}

static void WaitChild()
{
   // Wait till child (i.e. ROOT) is finished.

   int status;

   do {
      while (waitpid(gChildpid, &status, WUNTRACED) < 0) {
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
         kill(gChildpid, SIGCONT);       // if parent wakes up, wake up child
      }
   } while (WIFSTOPPED(status));

   exit(0);
}

static void PrintUsage()
{
   fprintf(stderr, kCommandLineOptionsHelp);
}

int main(int argc, char **argv)
{
   char **argvv;
   char  arg0[kMAXPATHLEN];

#ifdef ROOTPREFIX
   if (getenv("ROOTIGNOREPREFIX")) {
#endif
   // Try to set ROOTSYS depending on pathname of the executable
   SetRootSys();

   if (!getenv("ROOTSYS")) {
      fprintf(stderr, "%s: ROOTSYS not set. Set it before trying to run %s.\n",
              argv[0], argv[0]);
      return 1;
   }
#ifdef ROOTPREFIX
   }
#endif

   // In batch mode don't show splash screen, idem for no logo mode,
   // in about mode show always splash screen
   int i;
   for (i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-?") || !strncmp(argv[i], "-h", 2) ||
          !strncmp(argv[i], "--help", 6)) {
         PrintUsage();
         return 0;
      }
   }

   // Ignore SIGINT and SIGQUIT. Install handler for SIGUSR1.

   struct sigaction ignore, actTerm,
      saveintr, savequit, saveterm;

#if defined(__sun) && !defined(__i386) && !defined(__SVR4)
   ignore.sa_handler = (void (*)())SIG_IGN;
#elif defined(__sun) && defined(__SVR4)
   ignore.sa_handler = (void (*)(int))SIG_IGN;
#else
   ignore.sa_handler = SIG_IGN;
#endif
   sigemptyset(&ignore.sa_mask);
   ignore.sa_flags = 0;

   actTerm = ignore;
#if defined(__sun) && !defined(__i386) && !defined(__SVR4)
   actTerm.sa_handler = (void (*)())SigTerm;
#elif defined(__sun) && defined(__SVR4)
   actTerm.sa_handler = SigTerm;
#elif (defined(__sgi) && !defined(__KCC)) || defined(__Lynx__)
#   if defined(IRIX64) || (__GNUG__>=3)
   actTerm.sa_handler = SigTerm;
#   else
   actTerm.sa_handler = (void (*)(...))SigTerm;
#   endif
#else
   actTerm.sa_handler = SigTerm;
#endif
   sigaction(SIGINT,  &ignore, &saveintr);
   sigaction(SIGQUIT, &ignore, &savequit);
   sigaction(SIGTERM, &actTerm, &saveterm);

   // Create child...

   if ((gChildpid = fork()) < 0) {
      fprintf(stderr, "%s: error forking child\n", argv[0]);
      return 1;
   } else if (gChildpid > 0) {
      WaitChild();
   }

   // Continue with child...

   // Restore original signal actions
   sigaction(SIGINT,  &saveintr, 0);
   sigaction(SIGQUIT, &savequit, 0);
   sigaction(SIGTERM, &saveterm, 0);

   // Child is going to overlay itself with the actual ROOT module...

   // Build argv vector
   argvv = new char* [argc+1];
#ifdef ROOTBINDIR
   if (getenv("ROOTIGNOREPREFIX"))
#endif
      snprintf(arg0, sizeof(arg0), "%s/bin/%s", getenv("ROOTSYS"), ROOTBINARY);
#ifdef ROOTBINDIR
   else
      snprintf(arg0, sizeof(arg0), "%s/%s", ROOTBINDIR, ROOTBINARY);
#endif
   argvv[0] = arg0;

   for (i = 1; i < argc; i++)
      argvv[i] = argv[i];
   argvv[i] = 0;

#ifndef IS_RPATH_BUILD
   // Make sure library path is set
   SetLibraryPath();
#endif

   // Execute actual ROOT module
   execv(arg0, argvv);

   // Exec failed
   fprintf(stderr, "%s: can't start ROOT -- check that %s exists!\n",
           argv[0], arg0);

   delete [] argvv;

   return 1;
}
